from base import BaseKFACInfluenceModule
import torch
import tqdm

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

    
