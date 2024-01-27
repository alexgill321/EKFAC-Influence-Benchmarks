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
        self.cov_loader = cov_loader
        self._bwd_handles = []
        self._fwd_handles = []

        self.layer_modules = [
            self._get_module_from_name(self.model,layer) for layer in layers
        ] 

        self.state = {layer: {} for layer in self.layer_modules}


        if cov_loader is None:
            self.cov_loader = train_loader

        self._layer_hooks()

        self._compute_kfac_params()

        self._invert_covs()
        
    def inverse_hvp(self, vec):
        for layer in self.layer_modules:
            yield torch.mm(self.state[layer]['sinv'], torch.mm(vec, self.state[layer]['ainv']))

    def _compute_kfac_params(self):
        cov_batched = tqdm.tqdm(self.cov_loader, total=len(self.cov_loader), desc="Calculating Covariances")

        for batch in cov_batched:
            loss = self._loss_pseudograd(batch)
            for l in loss:
                l.backward(retain_graph=True)
                self._update_covs()
                self.model.zero_grad()

        # May have to change based on intended batching
        for layer in self.layer_modules:
            self.state[layer]['acov'] = self.state[layer]['acov'].div(len(self.cov_loader))
            self.state[layer]['scov'] = self.state[layer]['scov'].div(len(self.cov_loader))
    
    def _invert_covs(self):
        for layer in self.layer_modules:
            acov = self.state[layer]['acov']
            gycov = self.state[layer]['scov']

            # Invert the covariances
            self.state[layer]['ainv'] = (self.state[layer]['acov'] + 
                                       self.damp * torch.eye(acov.shape[0])).inverse().to(self.device)
            self.state[layer]['sinv'] = (self.state[layer]['scov'] + 
                                        self.damp * torch.eye(gycov.shape[0])).inverse().to(self.device)

    def _layer_hooks(self):
        for layer in self.layer_modules:
            mod_class = layer.__class__.__name__
            if mod_class in ['Linear']:
                handle = layer.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = layer.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)

    def _update_covs(self):
        for layer in self.layer_modules:
            x = self.state[layer]['x']
            gy = self.state[layer]['gy']

            x = x.data.t()
            gy = gy.data.t()

            if layer.bias is not None:
                ones = torch.ones_like(x[:1])
                x = torch.cat([x, ones], dim=0)
            
            self._calc_covs(layer, x, gy)
    
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

    
