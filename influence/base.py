import abc
from tqdm import tqdm

import torch
import numpy as np
from itertools import chain
from torch import Tensor, nn
from torch.utils import data
from functools import reduce
from typing import Any, List, Optional, Union
from tqdm import tqdm
from accelerate import Accelerator

def _set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        _set_attr(getattr(obj, names[0]), names[1:], val)

def _del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_attr(getattr(obj, names[0]), names[1:])

def print_memory_usage():
    for device in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        print(f"cuda:{device}: Allocated memory: {allocated:.2f} GB")
        print(f"cuda:{device}: Reserved memory: {reserved:.2f} GB")
        print(f"summary: {torch.cuda.memory_summary(device)}")

def get_memory_usage():
    allocated_memory = {}

    # Iterate through each available CUDA device
    for device_id in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{device_id}")
        allocated_memory[f"Cuda:{device_id} Mem Allocated (GB)"] = f"{torch.cuda.memory_allocated(device) / (1024 ** 3):.2f}"

    return allocated_memory

class BaseInfluenceObjective(abc.ABC):
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
    
class KFACBaseInfluenceObjective(BaseInfluenceObjective):
    def pseudograd_loss(self, model: nn.Module, batch: Any, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

class BaseInfluenceModule(abc.ABC):
    def __init__(
            self,
            model: nn.Module,
            objective: BaseInfluenceObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            accelerator: Optional[Accelerator] = None
    ):
        self.model = model
        self.device = device
        self.accelerator = accelerator

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
        return self._loss_grads(train_idxs, train=True)
    
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

        for grad_z in self._loss_grad_loader_wrapper(batch_size=1, subset=train_idxs, train=True):

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
            vec.append(p.view(-1).detach().to("cuda:1") if torch.cuda.device_count() > 1 else p.view(-1).detach())
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
        loss_fn = self.objective.train_loss if train else self.objective.test_loss
        for batch, _ in self._loader_wrapper(train=train, **kwargs):
            loss = loss_fn(self.model, batch=batch)
            params = self._model_params(with_names=False)
            grads = torch.autograd.grad(loss, params)
            yield self._flatten_params_like(grads)
            

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

        if self.accelerator is not None:
            new_loader = self.accelerator.prepare_data_loader(new_loader)

        data_left = len(dataset)
        for batch in new_loader:
            if self.accelerator is None:
                batch = self._transfer_to_device(batch)
            size = min(batch_size, data_left)  # deduce batch size
            yield batch, size
            data_left -= size

    def _get_module_from_name(self, model, layer_name) -> Any:
        return reduce(getattr, layer_name.split("."), model)
    
class BaseLayerInfluenceModule(BaseInfluenceModule):
    def __init__(
        self,
        model: nn.Module,
        objective: BaseInfluenceObjective,
        train_loader: data.DataLoader,
        test_loader: data.DataLoader,
        device: torch.device,
        layers: Union[str, List[str]],
        accelerator: Optional[Accelerator] = None,
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            accelerator=accelerator
        )
        self._bwd_handles = []
        self._fwd_handles = []

        self.layer_names = layers
        self.layer_modules = [
            self._get_module_from_name(self.model, layer) for layer in layers
        ]

        self.layer_param_names = {
            layer_name: tuple(name for name, _ in self._layer_params(layer)) for layer, layer_name in zip(self.layer_modules, self.layer_names)
        }
        self.layer_param_shapes = {
            layer_name: tuple(param.shape for _, param in self._layer_params(layer)) for layer, layer_name in zip(self.layer_modules, self.layer_names)
        }

        self.state = {layer: {} for layer in set(chain(self.layer_modules, self.layer_names))}

    @abc.abstractmethod
    def inverse_hvp(self, vec):
        raise NotImplementedError()
    
    def influences(self,
                   train_idxs: List[int],
                   test_idxs: List[int]
                   ) -> Tensor:

        ihvps = self._compute_ihvps(test_idxs)
        torch.cuda.empty_cache()
        scores = {}

        training_srcs = tqdm(
            self._loss_grad_loader_wrapper(train=True, subset=train_idxs, batch_size=1),
            total=len(train_idxs), 
            desc="Calculating Training Loss Grads"
            )
        for grad in training_srcs:
            grads = self._reshape_like_params(grad) 
            training_srcs.set_postfix(get_memory_usage())

            # get batch of 15 ihvps
            for layer_name, layer in zip(self.layer_names, self.layer_modules):
                layer_grad = self._flatten_params_like(self._reshape_like_layer_params(grads, layer, layer_name)).detach().to("cpu")
                if layer_name not in scores:
                    scores[layer_name] = (layer_grad @ ihvps[layer_name]).view(-1, 1).detach().to("cpu")
                else:
                    scores[layer_name] = torch.cat([scores[layer_name], (layer_grad @ ihvps[layer_name]).view(-1, 1)], dim=1)
         
        return scores
    
    def _compute_ihvps(self, test_idxs: List[int]) -> torch.Tensor:
        ihvps = {}
        queries = tqdm(
            self._loss_grad_loader_wrapper(train=False, subset=test_idxs, batch_size=1), 
            total=len(test_idxs), 
            desc="Calculating IHVPS"
            )
        
        for grad_q in queries:
            queries.set_postfix(get_memory_usage())
            ihvp = self.inverse_hvp(grad_q) # Per layer IHVPs for a single query
            for layer in self.layer_names:
                if layer not in ihvps:
                    ihvps[layer] = ihvp[layer].view(-1, 1).detach().to("cpu") #Dictionary of matrices for each layer (n_queries x (m *n))
                else:
                    ihvps[layer] = torch.cat([ihvps[layer], ihvp[layer].view(-1, 1).detach().to("cpu")], dim=1) #Dictionary of matrices for each layer (n_queries x (m *n))
        return ihvps

    def _layer_hooks(self):
        for layer in self.layer_modules:
            handle = layer.register_forward_pre_hook(self._save_input)
            self._fwd_handles.append(handle)
            handle = layer.register_full_backward_hook(self._save_grad_output)
            self._bwd_handles.append(handle)
        
    def _save_input(self, layer, inp):
        self.state[layer]['x'] = inp[0]
    
    def _save_grad_output(self, layer, grad_input, grad_output):
        self.state[layer]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _reshape_like_layers_params(self, vec):
        params = self._reshape_like_params(vec)
        layer_grads = []
        for layer_name, layer in zip(self.layer_names, self.layer_modules):
            layer_grads += self._reshape_like_layer_params(params, layer, layer_name)
        return layer_grads
    
    def _reshape_like_layer_params(self, params, layer, layer_name):
            if layer.__class__.__name__ == 'Linear':
                layer_grad = params[self.params_names.index(layer_name + '.weight')]
                
                if layer.bias is not None:
                    layer_grad = torch.cat([layer_grad, params[self.params_names.index(layer_name + '.bias')].view(-1,1)], dim=1)
                
                return layer_grad
            else:
                raise NotImplementedError()
    
    def _layer_make_functional(self, layer, layer_name):
        assert not self.is_layer_functional
        params = tuple(p.detach().requires_grad_() for p in self._layer_params(layer, False))

        for name in self.layer_param_names[layer_name]:
            _del_attr(layer, name.split('.'))

        self.is_layer_functional = True

        return params
    
    def _reshape_like_layers(self, vec):
        params = self._reshape_like_params(vec)
        layer_grads = {}
        for layer_name, layer in zip(self.layer_names, self.layer_modules):
            if layer.__class__.__name__ == 'Linear':
                layer_grad = params[self.params_names.index(layer_name + '.weight')]

                if layer.bias is not None:
                    layer_grad = torch.cat([layer_grad, params[self.params_names.index(layer_name + '.bias')].view(-1,1)], dim=1)
                layer_grads[layer_name] = layer_grad
        return layer_grads
    
    def _reshape_like_layer(self, vec, layer):
        pointer = 0
        layer_tensors = []

        for dim in self.layer_param_shapes[layer]:
            num_param = dim.numel()
            layer_tensors.append(vec[pointer: pointer + num_param].view(dim))
            pointer += num_param
        
        return tuple(layer_tensors)
            
    def _reinsert_layer_params(self, layer, layer_name, layer_params, register=False):
            for name, param in zip(self.layer_param_names[layer_name], layer_params):
                _set_attr(layer, name.split('.'), torch.nn.Parameter(param) if register else param)
            self.is_layer_functional = not register

    def _layer_params(self, layer, with_names=True):
        return tuple((name, p) if with_names else p for name, p in layer.named_parameters() if p.requires_grad)

class BaseKFACInfluenceModule(BaseLayerInfluenceModule):
    def __init__(
            self,
            model: nn.Module,
            objective: KFACBaseInfluenceObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            layers: Union[str, List[str]],
            accelerator: Optional[Accelerator] = None,
            cov_loader: Optional[data.DataLoader] = None,
            n_samples: int = 1,
            damp: float = 1e-6,
            seed: int = 42
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            accelerator=accelerator,
            layers=layers
        )
        self.damp = damp
        self.n_samples = n_samples
        self.cov_loader = cov_loader

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        if cov_loader is None:
            self.cov_loader = train_loader

        self.compute_kfac_params()

    @abc.abstractmethod
    def compute_kfac_params(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def inverse_hvp(self, vec):
        raise NotImplementedError()
    
    def _update_covs(self):
        for layer_name, layer in zip(self.layer_names, self.layer_modules):
            with torch.no_grad():
                x = self.state[layer]['x'].detach().to("cuda:1") if torch.cuda.device_count() > 1 else self.state[layer]['x'].detach()
                gy = self.state[layer]['gy'].detach().to("cuda:1") if torch.cuda.device_count() > 1 else self.state[layer]['gy'].detach()
                
                if x.dim() == 2:
                    x = x.data.t()
                    gy = gy.data.t()

                    if layer.bias is not None:
                        ones = torch.ones_like(x[:1])
                        x = torch.cat([x, ones], dim=0)

                """
                x dimensions (before permute):

                0: batch size
                1: sequence length
                2: hidden size
                """
                if x.dim() == 3:
                    x = x.permute(0, 2, 1)
                    gy = gy.permute(0, 2, 1)
                    if layer.bias is not None:
                        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

                self._calc_covs(layer_name, x, gy)
    
    def _calc_covs(self, layer, x, gy):
        if x.dim == 2:
            acov = x.mm(x.t()) / x.shape[1]
            scov = gy.mm(gy.t()) / gy.shape[1]
            if 'acov' not in self.state[layer]:
                self.state[layer]['acov'] = acov
            else:
                self.state[layer]['acov'].add_(acov)
            if 'scov' not in self.state[layer]:
                self.state[layer]['scov'] = scov
            else:
                self.state[layer]['scov'].add_(scov)
                
        elif x.dim() == 3:
            if 'acov' not in self.state[layer]:
                self.state[layer]['acov'] = torch.sum(torch.matmul(x, x.transpose(1, 2))/x.shape[2], dim=0)/x.shape[0]
            else:
                self.state[layer]['acov'].add_(torch.sum(torch.matmul(x, x.transpose(1, 2))/x.shape[2], dim=0)/x.shape[0])
            if 'scov' not in self.state[layer]:
                self.state[layer]['scov'] = torch.sum(torch.matmul(gy, gy.transpose(1, 2))/gy.shape[2], dim=0)/gy.shape[0]
            else:
                self.state[layer]['scov'].add_(torch.sum(torch.matmul(gy, gy.transpose(1, 2))/gy.shape[2], dim=0)/gy.shape[0])

