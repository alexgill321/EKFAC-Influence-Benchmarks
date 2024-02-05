import logging
from typing import List, Union

import numpy as np
from influence.base import BaseKFACInfluenceModule, BaseLayerInfluenceModule, BaseInfluenceObjective
import torch
import tqdm
import torch.nn as nn
from torch.utils import data


import sys
sys.path.append('c:\\Users\\alexg\\Documents\\GitHub\\EKFAC-Influence-Benchmarks')

class KFACInfluenceModule(BaseKFACInfluenceModule):  
    def inverse_hvp(self, vec):
        layer_grads = self._reshape_like_layers(vec)

        ihvps = {}
        for layer in self.layer_names:
            ihvps[layer] = torch.mm(self.state[layer]['sinv'], torch.mm(layer_grads[layer], self.state[layer]['ainv']))
        
        return ihvps
            
    def compute_kfac_params(self):
        self._layer_hooks()

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
        
        self._invert_covs()
    
    def _invert_covs(self):
        for layer in self.layer_names:
            acov = self.state[layer]['acov']
            scov = self.state[layer]['scov']

            # Invert the covariances
            self.state[layer]['ainv'] = (acov + self.damp * torch.eye(acov.shape[0]).to(self.device)).inverse()
            self.state[layer]['sinv'] = (scov + self.damp * torch.eye(scov.shape[0]).to(self.device)).inverse()

    
class EKFACInfluenceModule(BaseKFACInfluenceModule):
    def inverse_hvp(self, vec):
        layer_grads = self._reshape_like_layers(vec)
        
        ihvps = {}
        for layer in self.layer_names:
            qs = self.state[layer]['qs']
            qa = self.state[layer]['qa']
            diag = self.state[layer]['diag']
            v_kfe = qs.t().mm(layer_grads[layer]).mm(qa)
            ihvps[layer] = qs.mm(v_kfe.div(diag.view(*v_kfe.size()) + self.damp)).mm(qa.t())

        return ihvps
            
    def compute_kfac_params(self):
        self._layer_hooks()

        cov_batched = tqdm.tqdm(self.cov_loader, total=len(self.cov_loader), desc="Calculating Covariances")

        for batch in cov_batched:
            loss = self._loss_pseudograd(batch, n_samples=self.n_samples, generator=self.generator)
            for l in loss:
                l.backward(retain_graph=True)
                self._update_covs()
                self.model.zero_grad()

        # May have to change based on intended batching
        for layer in self.layer_names:
            self.state[layer]['acov'] = self.state[layer]['acov'].div(len(self.cov_loader)*self.n_samples)
            self.state[layer]['scov'] = self.state[layer]['scov'].div(len(self.cov_loader)*self.n_samples)
            _, self.state[layer]['qa'] = torch.linalg.eigh(self.state[layer]['acov'])
            _, self.state[layer]['qs'] = torch.linalg.eigh(self.state[layer]['scov'])

        self._compute_ekfac_diags()

    def _compute_ekfac_diags(self):
        cov_batched = tqdm.tqdm(
            self.cov_loader, 
            total=len(self.cov_loader), 
            desc="Calculating EKFAC Diagonals"
            )
        
        for batch in cov_batched:
            loss = self._loss_pseudograd(batch, n_samples=self.n_samples, generator=self.generator)
            for l in loss:
                l.backward(retain_graph=True)
                self._update_diags()
                self.model.zero_grad()

        for layer in self.layer_names:
            self.state[layer]['diag'] = self.state[layer]['diag'].div(len(self.cov_loader)*self.n_samples)

    def _update_diags(self):
        for layer_name, layer in zip(self.layer_names, self.layer_modules):
            x = self.state[layer]['x']
            gy = self.state[layer]['gy']

            qa = self.state[layer_name]['qa']
            qs = self.state[layer_name]['qs']

            if layer.bias is not None:
                x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

            diag = (gy.mm(qs).t() ** 2).mm(x.mm(qa) ** 2).view(-1)
            if 'diag' not in self.state[layer_name]:
                self.state[layer_name]['diag'] = diag
            else:
                self.state[layer_name]['diag'].add_(diag)
            
class PBRFInfluenceModule(BaseLayerInfluenceModule):
    def __init__(
            self,
            model: nn.Module,
            objective: BaseInfluenceObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            layers: Union[str, List[str]],
            check_eigvals: bool = False
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            layers=layers
        )
        self.damp = damp
        self.is_layer_functional = False

        layer = self.layer_modules[0]
        layer_name = self.layer_names[0]

        # For now only single layer support
        params = self._layer_make_functional(layer, layer_name)
        flat_params = self._flatten_params_like(params)
        d = flat_params.shape[0]

        gnh = 0.0

        for batch, batch_size in tqdm.tqdm(self._loader_wrapper(train=True, batch_size=1), total=len(self.train_loader.dataset), desc="Estimating Hessian"):
                def layer_f(y):
                    return self.objective.train_loss_on_outputs(y, batch)
                
                def layer_out_f(theta_l):
                    self._reinsert_layer_params(layer, layer_name, self._reshape_like_layer(theta_l, layer_name))
                    return self.objective.train_outputs(self.model, batch)

                self._reinsert_layer_params(layer, layer_name, self._reshape_like_layer(flat_params, layer_name))
                outputs = self.objective.train_outputs(self.model, batch)
                o = outputs.shape[1]

                hess_batch = torch.autograd.functional.hessian(layer_f, outputs, strict=True).reshape(o,o)
                jac_batch = torch.autograd.functional.jacobian(layer_out_f, flat_params, strict=True).squeeze(0)

                gnh_batch = jac_batch.t().mm(hess_batch.mm(jac_batch))
                gnh += gnh_batch * batch_size

        with torch.no_grad():
            self._reinsert_layer_params(layer, layer_name, self._reshape_like_layer(flat_params, layer_name), register=True)
            gnh = gnh / len(self.train_loader.dataset)
            gnh = gnh + damp * torch.eye(d, device=self.device)

            if check_eigvals:
                eigvals = np.linalg.eigvalsh(gnh.cpu().numpy())
                logging.info("hessian min eigval %f", np.min(eigvals).item())
                logging.info("hessian max eigval %f", np.max(eigvals).item())
                if not bool(np.all(eigvals >= 0)):
                    raise ValueError()
            
            self.inverse_gnh = torch.inverse(gnh)
    
    def inverse_hvp(self, vec):
        layer_grads = self._reshape_like_layers(vec)
        ihvps = {}

        for layer in self.layer_names:
            ihvps[layer] = self.inverse_gnh @ layer_grads[layer].view(-1)

        return ihvps

