import abc
from typing import Any, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils import data
from functools import reduce

class BaseObjective(abc.ABC):
    @abc.abstractmethod
    def train_outputs(self, model: nn.Module, batch: Any) -> Any:
        """
        Returns the outputs of the model on the given batch of training data.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def train_loss_on_outputs(self, outputs: torch.Tensor, batch: Any) -> torch.Tensor:
        """
        Returns the loss on the given batch of training data.
        """
        raise NotImplementedError()
    
    def train_loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Returns the **mean**-reduced regularized loss of a model over a batch of data.

        This method should not be overridden for most use cases. By default, torch-influence
        takes and expects the overall training loss to be::

            outputs = train_outputs(model, batch)
            loss = train_loss_on_outputs(outputs, batch) + train_regularization(params)

        Args:
            model: the model.
            params: a flattened vector of the model's parameters.
            batch: a batch of training data.

        Returns:
            the training loss over the batch.
        """

        outputs = self.train_outputs(model, batch)
        return self.train_loss_on_outputs(outputs, batch)
    
    @abc.abstractmethod
    def test_loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """
        Returns the loss on the given batch of test data.
        """
        raise NotImplementedError()
    

class BaseInfluenceModule(abc.ABC):
    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device
    ):
        model.eval()
        self.model = model.to(device)
        self.device = device

        self.is_model_functional = False
        self.params_names = tuple(name for name, _ in self._model_params())
        self.params_shape = tuple(p.shape for _, p in self._model_params())

        self.objective = objective
        self.train_loader = train_loader
        self.test_loader = test_loader

    @abc.abstractmethod
    def inverse_hvp(self, vec: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError()
    
    def train_loss_grad(self, train_idxs: List[int]) -> torch.Tensor:
        """
        Returns the gradient of the training loss with respect to the model parameters.
        """
        return self._loss_grad(train_idxs, train=True)
    
    def test_loss_grads(self, test_idxs: List[int]) -> torch.Tensor:
        """
        Returns the gradient of the test loss with respect to the model parameters.
        """
        return self._loss_grads(test_idxs, train=False)
    
    def query_grads(self, test_idxs: List[int]) -> torch.Tensor:
        
        # TODO: RANK 32 approximations for query grads??
        ihvps = []
        for grad_q in self.test_loss_grads(test_idxs):
            ihvps.append(self.inverse_hvp(grad_q))
                                 
        return torch.cat(ihvps, dim=0)
    
    def influences(self,
                   train_idxs: List[int],
                   test_idxs: List[int],
                   num_samples: Optional[int] = None
                   ) -> torch.Tensor:
    
        grads_q = self.query_grads(test_idxs)
        scores = []

        for grad_z, _ in self._loss_grad_loader_wrapper(batch_size=1, subset=train_idxs, train=True):
            s = grad_z.mm(grads_q)
            scores.append(s)
        
        return torch.tensor(scores) / len(self.train_loader.dataset)
    
    ### Private Methods ###
    def _model_params(self, with_names=True):
        assert not self.is_model_functional
        return tuple((name, p) if with_names else p for name, p in self.model.named_parameters() if p.requires_grad)
    
    def _flatten_params_like(self, params_like):
        vec = []
        for p in params_like:
            vec.append(p.view(-1))
        return torch.cat(vec)

    def _reshape_like_params(self, vec):
        pointer = 0
        split_tensors = []
        for dim in self.params_shape:
            num_param = dim.numel()
            split_tensors.append(vec[pointer: pointer + num_param].view(dim))
            pointer += num_param
        return tuple(split_tensors)
    
    def _transfer_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (tuple, list)):
            return type(batch)(self._transfer_to_device(x) for x in batch)
        elif isinstance(batch, dict):
            return {k: self._transfer_to_device(x) for k, x in batch.items()}
        else:
            raise NotImplementedError()

    def _loss_grads(self, idxs, train):
        grads = None
        for grad in self._loss_grad_loader_wrapper(batch_size=1, subset=idxs, train=train):
            grads = grad.view(1, -1) if grads is None else torch.cat((grads, grad.view(1, -1)), dim=0)
        
        return grads
    
    def _loss_grad_loader_wrapper(self, train, **kwargs):

        for batch, _ in self._loader_wrapper(train=train, **kwargs):
            loss_fn = self.objective.train_loss if train else self.objective.test_loss
            loss = loss_fn(self.model, batch=batch)
            yield self._flatten_params_like(torch.autograd.grad(loss, self._model_params(with_names=False)))


    def _loader_wrapper(self, train, batch_size=None, subset=None, sample_n_batches=-1):
        loader = self.train_loader if train else self.test_loader
        batch_size = loader.batch_size if (batch_size is None) else batch_size
        
        if subset is None:
            dataset = loader.dataset
        else:
            subset = np.array(subset)
            if len(subset.shape) != 1 or len(np.unique(subset)) != len(subset):
                raise ValueError()
            if np.any((subset < 0) | (subset >= len(loader.dataset))):
                raise IndexError()
            dataset = data.Subset(loader.dataset, indices=subset)

        if sample_n_batches > 0:
            num_samples = sample_n_batches * batch_size
            sampler = data.RandomSampler(data_source=dataset, replacement=True, num_samples=num_samples)
        else:
            sampler = None
        
        new_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            batch_sampler=None,
            collate_fn=loader.collate_fn,
            num_workers=loader.num_workers,
            worker_init_fn=loader.worker_init_fn,
        )

        data_left = len(dataset)
        for batch in new_loader:
            batch = self._transfer_to_device(batch)
            size = min(batch_size, data_left)  # deduce batch size
            yield batch, size
            data_left -= size

    def _loss_pseudograd(self, batch, n_samples=1):
        outputs = self.objective.train_outputs(self.model, batch)
        output_probs = torch.softmax(outputs, dim=-1)
        samples = torch.multinomial(output_probs, num_samples=n_samples, replacement=True)
        for s in samples.t():
            inputs = batch[0].clone()
            sampled_batch = [inputs, s]
            yield self.objective.train_loss_on_outputs(outputs, sampled_batch)

    def _get_module_from_name(self, model, layer_name) -> Any:
        return reduce(getattr, layer_name.split("."), model)


