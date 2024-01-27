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
    
    def train_loss(self, model: nn.Module, params: torch.Tensor, batch: Any) -> torch.Tensor:
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
        return self.train_loss_on_outputs(outputs, batch) + self.train_regularization(params)
    
    @abc.abstractmethod
    def test_loss(self, model: nn.Module, params: torch.Tensor, batch: Any) -> torch.Tensor:
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
    
    def test_loss_grad(self, test_idxs: List[int]) -> torch.Tensor:
        """
        Returns the gradient of the test loss with respect to the model parameters.
        """
        return self._loss_grad(test_idxs, train=False)
    
    def stest(self, test_idxs: List[int]) -> torch.Tensor:
        
        return self.inverse_hvp(self.test_loss_grad(test_idxs)).to(self.device)
    
    def influences(self,
                   train_idxs: List[int],
                   test_idxs: List[int],
                   num_samples: Optional[int] = None
                   ) -> torch.Tensor:
    
        stest = self.stest(test_idxs)
        scores = []

        for grad_z, _ in self.loss_grad_loader_wrapper(batch_size=1, subset=train_idxs, train=True):
            s = grad_z.mm(stest)
            scores.append(s)
        
        return torch.tensor(scores) / len(self.train_loader.dataset)
    
    ### Private Methods ###

    def _loss_pseudograd(self, batch, n_samples=2):
        outputs = self.objective.train_outputs(self.model, batch)
        output_probs = torch.softmax(outputs, dim=-1)
        samples = torch.multinomial(output_probs, num_samples=n_samples, replacement=True)
        for s in samples.t():
            inputs = batch[0].clone()
            sampled_batch = [inputs, s]
            yield self.objective.train_loss_on_outputs(outputs, sampled_batch)

    def _get_module_from_name(self, model, layer_name) -> Any:
        return reduce(getattr, layer_name.split("."), model)


