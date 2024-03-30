import logging
from typing import List, Union

import numpy as np
from influence.base import BaseKFACInfluenceModule, BaseLayerInfluenceModule, BaseInfluenceObjective, print_memory_usage
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils import data

class KFACInfluenceModule(BaseKFACInfluenceModule):  
    def inverse_hvp(self, vec):
        layer_grads = self._reshape_like_layers(vec)

        ihvps = {}
        for layer_name in self.layer_names:
            ihvps[layer_name] = torch.mm(self.state[layer_name]['sinv'], torch.mm(layer_grads[layer_name], self.state[layer_name]['ainv']))
        
        return ihvps
            
    def compute_kfac_params(self):
        self._layer_hooks()

        cov_batched = tqdm(self.cov_loader, total=len(self.cov_loader), desc="Calculating Covariances")
        for batch in cov_batched:
            loss = self.objective.pseudograd_loss(self.model,batch, n_samples=self.n_samples)
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
        for layer_name in self.layer_names:
            qs = self.state[layer_name]['qs']
            qa = self.state[layer_name]['qa']
            diag = self.state[layer_name]['diag']
            v_kfe = qs.t().mm(layer_grads[layer_name]).mm(qa)
            ihvps[layer_name] = qs.mm(v_kfe.div(diag.view(*v_kfe.size()) + self.damp)).mm(qa.t())

        return ihvps
            
    def compute_kfac_params(self):
        self._layer_hooks()

        cov_batched = tqdm(self.cov_loader, total=len(self.cov_loader), desc="Calculating Covariances")

        for batch in cov_batched:
            cov_batched.set_postfix({"Allocated memory": f"{torch.cuda.memory_allocated(self.device) / (1024 ** 3):.2f} GB", "Batch size": batch[0].shape})
            losses = self.objective.pseudograd_loss(self.model, batch, n_samples=self.n_samples, generator=self.generator)
            try:
                current_loss = next(losses)
            except StopIteration:
                # Handle case where cov_batched is empty
                current_loss = None
            while current_loss is not None:
                try:
                    next_loss = next(losses)
                    retain_graph = True
                except StopIteration:
                    next_loss = None
                    retain_graph = False
                current_loss.backward(retain_graph=retain_graph)
                self._update_covs()
                self.model.zero_grad()
                current_loss = next_loss

        # May have to change based on intended batching
        for layer in self.layer_names:
            self.state[layer]['acov'] = self.state[layer]['acov'].div(len(self.cov_loader)*self.n_samples)
            self.state[layer]['scov'] = self.state[layer]['scov'].div(len(self.cov_loader)*self.n_samples)
            _, self.state[layer]['qa'] = torch.linalg.eigh(self.state[layer]['acov'])
            _, self.state[layer]['qs'] = torch.linalg.eigh(self.state[layer]['scov'])

        self._compute_ekfac_diags()

    def _compute_ekfac_diags(self):
        cov_batched = tqdm(
            self.cov_loader, 
            total=len(self.cov_loader), 
            desc="Calculating EKFAC Diagonals"
            )
        
        for batch in cov_batched:
            losses = self.objective.pseudograd_loss(self.model, batch, n_samples=self.n_samples, generator=self.generator)
            try:
                current_loss = next(losses)
            except StopIteration:
                # Handle case where cov_batched is empty
                current_loss = None
            loss = self.objective.pseudograd_loss(self.model, batch, n_samples=self.n_samples, generator=self.generator)
            while current_loss is not None:
                try:
                    next_loss = next(losses)
                    retain_graph = True
                except StopIteration:
                    next_loss = None
                    retain_graph = False
                current_loss.backward(retain_graph=retain_graph)
                self._update_diags()
                self.model.zero_grad()
                current_loss = next_loss

        for layer in self.layer_names:
            self.state[layer]['diag'] = self.state[layer]['diag'].div(len(self.cov_loader)*self.n_samples)

    def _update_diags(self):
        for layer_name, layer in zip(self.layer_names, self.layer_modules):
            with torch.no_grad():
                x = self.state[layer]['x'].detach()
                gy = self.state[layer]['gy'].detach()

                qa = self.state[layer_name]['qa'].detach()
                qs = self.state[layer_name]['qs'].detach()

                if x.dim() == 2:
                    if layer.bias is not None:
                        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

                    diag = (gy.mm(qs).t() ** 2).mm(x.mm(qa) ** 2).view(-1)
                    if 'diag' not in self.state[layer_name]:
                        self.state[layer_name]['diag'] = diag
                    else:
                        self.state[layer_name]['diag'].add_(diag)

                elif x.dim() == 3:
                    x = x.permute(0, 2, 1)
                    if layer.bias is not None:
                        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
                    x = x.permute(0, 2, 1)
                
                    if 'diag' not in self.state[layer_name]:
                        self.state[layer_name]['diag'] = (torch.sum(torch.matmul(torch.matmul(gy, qs).permute(1, 2, 0),torch.matmul(x, qa).permute(1, 0, 2)), dim = 0) ** 2).view(-1)
                    else:
                        self.state[layer_name]['diag'].add_((torch.sum(torch.matmul(torch.matmul(gy, qs).permute(1, 2, 0),torch.matmul(x, qa).permute(1, 0, 2)), dim = 0) ** 2).view(-1))
            
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
            check_eigvals: bool = False,
            gnh: bool = False,
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
        self.gnh = gnh

        layer = self.layer_modules[0]
        layer_name = self.layer_names[0]

        # For now only single layer support
        params = self._layer_make_functional(layer, layer_name)
        flat_params = self._flatten_params_like(params)
        d = flat_params.shape[0]

        gnh = 0.0

        dataset_batched = tqdm(self._loader_wrapper(train=True), total=len(self.train_loader), desc="Estimating Hessian")

        for batch, batch_size in dataset_batched:
                def layer_f(y):
                    return self.objective.train_loss_on_outputs(outputs=y, batch=batch)
                
                def layer_f_hess(theta_l):
                    self._reinsert_layer_params(layer, layer_name, self._reshape_like_layer(theta_l, layer_name))
                    return self.objective.train_loss(self.model, batch)
                
                def layer_out_f(theta_l):
                    self._reinsert_layer_params(layer, layer_name, self._reshape_like_layer(theta_l, layer_name))
                    return self.objective.train_outputs(self.model, batch)
                
                if self.gnh:
                    self._reinsert_layer_params(layer, layer_name, self._reshape_like_layer(flat_params, layer_name))
                    outputs = self.objective.train_outputs(self.model, batch)

                    o = outputs.shape[1]
                    # print("dim before {}".format(outputs.shape))
                    # outputs, _ = torch.max(outputs, dim=2)
                    # print("dim after {}".format(outputs.shape))
                    # exit()

                    '''
                    option 1 : take hessian of loss of each token wrt output.
                    option 2 : reduce outputs from 217, 50000 to 500000
                    '''

                    ####hessian of token wrt loss

                    for token in range(outputs.size(1)):
                        outputs = outputs[:, token, :]

                        hess_batch = torch.autograd.functional.hessian(layer_f, outputs, vectorize=True).mean(0).mean(1)
                        print(hess_batch.shape)
                        exit()
                    hess_batch = torch.autograd.functional.hessian(layer_f, outputs, vectorize=True).mean(0).mean(1)
                    jac_batch = torch.autograd.functional.jacobian(layer_out_f, flat_params, vectorize=True).mean(0)

                    gnh_batch = jac_batch.t().mm(hess_batch.mm(jac_batch))
                    gnh += gnh_batch * batch_size
                else:
                    hess_batch = torch.autograd.functional.hessian(layer_f_hess, flat_params, strict=False, vectorize=True)
                    gnh += hess_batch * batch_size

        with torch.no_grad():
            self._reinsert_layer_params(layer, layer_name, self._reshape_like_layer(flat_params, layer_name), register=True)
            gnh = gnh / len(self.train_loader.dataset)
            gnh = gnh + self.damp * torch.eye(d, device=self.device)

            if check_eigvals:
                eigvals = np.linalg.eigvalsh(gnh.cpu().numpy())
                logging.info("hessian min eigval %f", np.min(eigvals).item())
                logging.info("hessian max eigval %f", np.max(eigvals).item())
                if not bool(np.all(eigvals >= 0)):
                    raise ValueError()
            
            self.inverse_gnh = torch.inverse(gnh)
    
    def inverse_hvp(self, vec):
        layer_grads = self._reshape_like_layers_params(vec)
        ihvps = {}

        for layer in self.layer_names:
            ihvps[layer] = self.inverse_gnh @ self._flatten_params_like(layer_grads[layer])

        return ihvps

