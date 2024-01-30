from itertools import chain

from torch._tensor import Tensor
from base import BaseInfluenceModule, BaseObjective
import torch
from torch import nn
from torch.utils import data
import tqdm
from typing import List, Union, Optional

import sys
sys.path.append('c:\\Users\\alexg\\Documents\\GitHub\\EKFAC-Influence-Benchmarks')

class KFACInfluenceModule(BaseInfluenceModule):
    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            layers: Union[str, List[str]],
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float = 1e-6,
            n_samples: int = 1,
            cov_loader: Optional[data.DataLoader] = None,
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp
        self.n_samples = n_samples
        self.cov_loader = cov_loader
        self._bwd_handles = []
        self._fwd_handles = []

        self.layer_names = layers
        self.layer_modules = [
            self._get_module_from_name(self.model, layer) for layer in layers
        ] 

        self.state = {layer: {} for layer in set(chain(self.layer_modules, self.layer_names))}


        if cov_loader is None:
            self.cov_loader = train_loader

        self._layer_hooks()

        self._compute_kfac_params()

        self._invert_covs()
        
    def inverse_hvp(self, vec):
        layer_grads = self._reshape_like_layers(vec)

        ihvps = {}
        for layer in self.layer_names:
            ihvps[layer] = torch.mm(self.state[layer]['sinv'], torch.mm(layer_grads[layer], self.state[layer]['ainv']))
        
        return ihvps
    
    def query_grads(self, test_idxs: List[int]) -> Tensor:
        ihvps = {}
        for grad_q in self.test_loss_grads(test_idxs):
            ihvp = self.inverse_hvp(grad_q)
            for layer in self.layer_names:
                ihvps[layer] = ihvp[layer].view(-1, 1) if layer not in ihvps else torch.cat([ihvps[layer], ihvp[layer].view(-1, 1)], dim=1)

        return ihvps
    
    def influences(self,
                   train_idxs: List[int],
                   test_idxs: List[int],
                   num_samples: Optional[int] = None
                   ) -> Tensor:
        
        grads_q = self.query_grads(test_idxs)
        scores = {}

        for grad_z in self._loss_grad_loader_wrapper(batch_size=1, subset=train_idxs, train=True):
            layer_grads = self._reshape_like_layers(grad_z)
            for layer in self.layer_names:
                layer_grad = layer_grads[layer].flatten()
                if layer not in scores:
                    scores[layer] = (layer_grad @ grads_q[layer]).view(-1, 1)
                else:
                    scores[layer] = torch.cat([scores[layer], (layer_grad @ grads_q[layer]).view(-1, 1)], dim=1)
                
        return scores
            

                

    def _compute_kfac_params(self):
        cov_batched = tqdm.tqdm(self.cov_loader, total=len(self.cov_loader), desc="Calculating Covariances")

        for batch in cov_batched:
            loss = self._loss_pseudograd(batch, n_samples=self.n_samples)
            for l in loss:
                l.backward(retain_graph=True)
                self._update_covs()
                self.model.zero_grad()

        # May have to change based on intended batching
        for layer in self.layer_names:
            self.state[layer]['acov'] = self.state[layer]['acov'].div(len(self.cov_loader)*self.n_samples)
            self.state[layer]['scov'] = self.state[layer]['scov'].div(len(self.cov_loader)*self.n_samples)
    
    def _invert_covs(self):
        for layer in self.layer_names:
            acov = self.state[layer]['acov']
            scov = self.state[layer]['scov']

            # Invert the covariances
            self.state[layer]['ainv'] = (acov + self.damp * torch.eye(acov.shape[0]).to(self.device)).inverse()
            self.state[layer]['sinv'] = (scov + self.damp * torch.eye(scov.shape[0]).to(self.device)).inverse()

    def _layer_hooks(self):
        for layer in self.layer_modules:
            mod_class = layer.__class__.__name__
            if mod_class in ['Linear']:
                handle = layer.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = layer.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)

    def _reshape_like_layers(self, vec):
        grads = self._reshape_like_params(vec)

        layer_grads = {}
        for layer_name, layer in zip(self.layer_names, self.layer_modules):
            if layer.__class__.__name__ == 'Linear':
                layer_grad = grads[self.params_names.index(layer_name + '.weight')]
                
                if layer.bias is not None:
                    layer_grad = torch.cat([layer_grad, grads[self.params_names.index(layer_name + '.bias')].view(-1, 1)], dim=1)

                layer_grads[layer_name] = layer_grad
        
        return layer_grads

    def _update_covs(self):
        for layer_name, layer in zip(self.layer_names, self.layer_modules):
            x = self.state[layer]['x']
            gy = self.state[layer]['gy']

            x = x.data.t()
            gy = gy.data.t()

            if layer.bias is not None:
                ones = torch.ones_like(x[:1])
                x = torch.cat([x, ones], dim=0)
            
            self._calc_covs(layer_name, x, gy)
    
    def _calc_covs(self, layer, x, gy):
        if 'acov' not in self.state[layer]:
            self.state[layer]['acov'] = x.mm(x.t()) / x.size(1)
        else:
            self.state[layer]['acov'].addmm_(x / x.size(1), x.t())

        if 'gycov' not in self.state[layer]:
            self.state[layer]['scov'] = gy.mm(gy.t()) / gy.size(1)
        else:
            self.state[layer]['scov'].addmm_(gy / gy.size(1), gy.t())
        
    def _save_input(self, layer, inp):
        self.state[layer]['x'] = inp[0]

    def _save_grad_output(self, layer, grad_input, grad_output):
        self.state[layer]['gy'] = grad_output[0] * grad_output[0].size(0)

    
