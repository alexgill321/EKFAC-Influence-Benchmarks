import copy

from tqdm import tqdm
import torch
from base import BaseKFACInfluenceModule, BasePBRFInfluenceModule



import sys

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


class IHVPInfluence(BasePBRFInfluenceModule):
    def inverse_hvp(self, vec):
        return self.inverse_hess @ vec
