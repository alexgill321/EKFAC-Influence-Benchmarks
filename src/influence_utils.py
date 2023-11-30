import torch
from torch.cuda.amp import autocast
from torch.optim.optimizer import Optimizer

import captum._utils.common as common
from captum.influence._core.influence import DataInfluence
from torch.nn import Module
from typing import Any, Dict, List, Union
from torch import Tensor
import torch.distributions as dist
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
import tqdm

class EKFAC(Optimizer):
    def __init__(self, net, eps):
        self.eps = eps
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self.net = net
        self.calc_act = True

        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_full_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)
        super(EKFAC, self).__init__(self.params, {})

    def step(self):
        for group in self.param_groups:
            mod = group['mod']
            x = self.state[mod]['x']
            gy = self.state[mod]['gy']

            # Computation of activation cov matrix for batch
            x = x.data.t()

            # Append column of ones to x if bias is not None
            if mod.bias is not None:
                ones = torch.ones_like(x[:1])
                x = torch.cat([x, ones], dim=0)

            # Computation of psuedograd of layer output cov matrix for batch
            gy = gy.data.t()

            self.calc_A(group, x)
            self.calc_S(group, gy)
        
    def calc_A(self, group, x):
        if self.calc_act:
            # Calculate covariance matrix for layer activations (A_{l})
            if 'A' not in group:
                group['A'] = torch.matmul(x, x.t())/float(x.shape[1])
                group['A_count'] = 1
            else:
                torch.add(group['A'], torch.matmul(x, x.t())/float(x.shape[1]), out=group['A'])
                group['A_count'] += 1
            
        
    def calc_S(self, group, gy):
            if 'S' not in group:
                group['S'] = torch.matmul(gy, gy.t())/float(gy.shape[1])
                group['S_count'] = 1
            else:
                torch.add(group['S'], torch.matmul(gy, gy.t())/float(gy.shape[1]), out=group['S'])
                group['S_count'] += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

class EKFACInfluence(DataInfluence):
    def __init__(
        self,
        module: Module,
        layers: Union[str, List[str]],
        influence_src_dataset: Dataset,
        model_id: str = "",
        batch_size: int = 1,
        cov_batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            module (Module): An instance of pytorch model. This model should define all of its
                layers as attributes of the model. The output of the model must be logits for the
                classification task.
            layers (Union[str, List[str]]): A list of layer names for which the influence will
                be computed.
            influence_src_dataset (torch.utils.data.Dataset): Pytorch dataset that is used to create
                a pytorch dataloader to iterate over the dataset. This is the dataset for which we will
                be seeking for influential instances. In most cases this is the training dataset.
            activation_dir (str): Path to the directory where the activation computations will be stored.
            model_id (str): The name/version of the model for which layer activations are being computed.
                Activations will be stored and loaded under the subdirectory with this name if provided.
            batch_size (int): Batch size for the dataloader used to iterate over the influence_src_dataset.
            query_batch_size (int): Batch size for the dataloader used to iterate over the query dataset.
            cov_batch_size (int): Batch size for the dataloader used to compute the activations.
            **kwargs: Any additional arguments that are necessary for specific implementations of the
                'DataInfluence' abstract class.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.module = module.to(self.device)
        self.layers = [layers] if isinstance(layers, str) else layers
        self.influence_src_dataset = influence_src_dataset
        self.model_id = model_id

        self.influence_src_dataloader = DataLoader(
            self.influence_src_dataset, batch_size=batch_size, shuffle=False
        )
        self.cov_src_dataloader = DataLoader(
            self.influence_src_dataset, batch_size=cov_batch_size, shuffle=True
        )
    
    def influence(
            self,
            query_dataset: Dataset,
            topk: int = 1,
            eps: float = 1e-5,
            **kwargs: Any,
        ) -> Dict:

        influences: Dict[str, Any] = {}
        query_grads: Dict[str, List[Tensor]] = {}
        influence_src_grads: Dict[str, List[Tensor]] = {}

        query_dataloader = DataLoader(
            query_dataset, batch_size=1, shuffle=False
        )

        layer_modules = [
            common._get_module_from_name(self.module, layer) for layer in self.layers
        ]

        G_list = self._compute_EKFAC_params()

        criterion = torch.nn.NLLLoss()
        print(f'Cacultating query gradients on trained model')
        for layer in layer_modules:
            query_grads[layer] = []
            influence_src_grads[layer] = []

        for _, (inputs, targets) in tqdm.tqdm(enumerate(query_dataloader), total=len(query_dataloader)):
            self.module.zero_grad()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.module(inputs)

            loss = criterion(outputs, targets.view(-1))
            loss.backward()

            for layer in layer_modules:
                Qa = G_list[layer]['Qa']
                Qs = G_list[layer]['Qs']
                eigenval_diag = G_list[layer]['lambda']
                if layer.bias is not None:
                    grad_bias = layer.bias.grad
                    grad_weights = layer.weight.grad
                    grad_bias = grad_bias.reshape(-1, 1)
                    grads = torch.cat((grad_weights, grad_bias), dim=1)
                else:
                    grads = layer.weight.grad

                ihvp = self.ihvp_mod(grads, Qa, Qs, eigenval_diag, eps)
                query_grads[layer].append(ihvp)

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        print(f'Cacultating training src gradients on trained model')
        for i, (inputs, targets) in tqdm.tqdm(enumerate(self.influence_src_dataloader), total=len(self.influence_src_dataloader)):
            self.module.zero_grad()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.module(inputs)
            loss = criterion(outputs, targets.view(-1))
            for single_loss in loss:
                single_loss.backward(retain_graph=True)

                for layer in layer_modules:
                    if layer.bias is not None:
                        grad_bias = layer.bias.grad
                        grad_weights = layer.weight.grad
                        grad_bias = grad_bias.reshape(-1, 1)
                        grads = torch.cat([grad_weights, grad_bias], dim=1)
                    else:
                        grads = layer.weight.grad
                    influence_src_grads[layer].append(torch.flatten(grads))

            # Calculate influences by batch to save memory
            for layer in layer_modules:
                query_grad_matrix = torch.stack(query_grads[layer], dim=0)
                influence_src_grad_matrix = torch.stack(influence_src_grads[layer], dim=0)
                tinf = torch.matmul(query_grad_matrix, torch.t(influence_src_grad_matrix))
                tinf = tinf.detach().cpu()
                if layer not in influences:
                    influences[layer] = tinf
                else:
                    influences[layer] = torch.cat((influences[layer], tinf), dim=1)
                influence_src_grads[layer] = []
                
        return influences
    
    def kfac_influence(
            self,
            query_dataset: Dataset,
            topk: int = 1,
            eps: float = 1e-5,
            **kwargs: Any,
        ) -> Dict:

        influences: Dict[str, Any] = {}
        query_grads: Dict[str, List[Tensor]] = {}
        influence_src_grads: Dict[str, List[Tensor]] = {}

        query_dataloader = DataLoader(
            query_dataset, batch_size=1, shuffle=False
        )

        layer_modules = [
            common._get_module_from_name(self.module, layer) for layer in self.layers
        ]

        G_list = self._compute_EKFAC_params()

        criterion = torch.nn.NLLLoss()
        print(f'Cacultating query gradients on trained model')
        for layer in layer_modules:
            query_grads[layer] = []
            influence_src_grads[layer] = []
        
        for _, (inputs, targets) in tqdm.tqdm(enumerate(query_dataloader), total=len(query_dataloader)):
            self.module.zero_grad()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.module(inputs)

            loss = criterion(outputs, targets.view(-1))
            loss.backward()

            for layer in layer_modules:
                A_inv = G_list[layer]['A']
                S_inv = G_list[layer]['S']

                if not torch.equal(torch.t(A), A):
                    print('A is not symmetric')
                if not torch.equal(torch.t(S), S):
                    print('S is not symmetric')
                if layer.bias is not None:
                    grad_bias = layer.bias.grad
                    grad_weights = layer.weight.grad
                    grad_bias = grad_bias.reshape(-1, 1)
                    grads = torch.cat((grad_weights, grad_bias), dim=1)
                else:
                    grads = layer.weight.grad

                ihvp = torch.matmul(S_inv, torch.matmul(grads, A_inv)).flatten()
                query_grads[layer].append(ihvp)

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        print(f'Cacultating training src gradients on trained model')
        for i, (inputs, targets) in tqdm.tqdm(enumerate(self.influence_src_dataloader), total=len(self.influence_src_dataloader)):
            self.module.zero_grad()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.module(inputs)
            loss = criterion(outputs, targets.view(-1))
            for single_loss in loss:
                single_loss.backward(retain_graph=True)

                for layer in layer_modules:
                    if layer.bias is not None:
                        grad_bias = layer.bias.grad
                        grad_weights = layer.weight.grad
                        grad_bias = grad_bias.reshape(-1, 1)
                        grads = torch.cat([grad_weights, grad_bias], dim=1)
                    else:
                        grads = layer.weight.grad
                    influence_src_grads[layer].append(torch.flatten(grads))

            # Calculate influences by batch to save memory
            for layer in layer_modules:
                query_grad_matrix = torch.stack(query_grads[layer], dim=0)
                influence_src_grad_matrix = torch.stack(influence_src_grads[layer], dim=0)
                tinf = torch.matmul(query_grad_matrix, torch.t(influence_src_grad_matrix))
                tinf = tinf.detach().cpu()
                if layer not in influences:
                    influences[layer] = tinf
                else:
                    influences[layer] = torch.cat((influences[layer], tinf), dim=1)
                influence_src_grads[layer] = []
    
    def ihvp(
            self,
            grads: List[Tensor],
            Qa: Tensor,
            Qs: Tensor,
            eigenval_diag: Tensor,
            eps: float = 1e-5,
    ) -> Tensor:
        t_Qa = torch.t(Qa)
        t_Qs = torch.t(Qs)
        mm1 = torch.matmul(grads, t_Qa)
        mm2 = torch.matmul(Qs, mm1)
        rec = torch.reciprocal(eigenval_diag + eps)
        rec_res = rec.reshape(mm2.shape[0], -1)
        mm3 = torch.matmul(mm2/rec_res, Qa)
        ihvp = torch.matmul(t_Qs, mm3)
        ihvp_flat = torch.flatten(ihvp)
        return ihvp_flat    

    def ihvp_mod(
            self,
            grads: List[Tensor],
            Qa: Tensor,
            Qs: Tensor,
            eigenval_diag: Tensor,
            eps: float = 1e-5,
    ) -> Tensor:
        t_Qa = torch.t(Qa)
        t_Qs = torch.t(Qs)

        inner_num = torch.matmul(Qs, torch.matmul(grads, t_Qa))
        inv_diag = 1/(eigenval_diag + eps)
        inner_prod = torch.div(inner_num, inv_diag)
        outer_prod = torch.matmul(t_Qs, torch.matmul(inner_prod, Qa))
        ihvp = torch.flatten(outer_prod)

        return ihvp

    def _compute_EKFAC_params(self, n_samples: int = 2):
        ekfac = EKFAC(self.module, 1e-5)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        for _, (input, _) in tqdm.tqdm(enumerate(self.cov_src_dataloader), total=len(self.cov_src_dataloader)):
            input = input.to(self.device)
            outputs = self.module(input)
            ekfac.calc_act = True
            output_probs = torch.softmax(outputs, dim=-1)
            distribution = dist.Categorical(output_probs)
            for _ in range(n_samples):
                samples = distribution.sample()
                loss = loss_fn(outputs, samples)
                loss.backward(retain_graph=True)
                ekfac.step()      
                
                self.module.zero_grad()
                ekfac.zero_grad()
                G_list = {}
                ekfac.calc_act = False
        # Compute average A and S
        for group in ekfac.param_groups:
            G_list[group['mod']] = {}
            with autocast():
                A = (group['A']/float(group['A_count'])).to('cpu')
                S = (group['S']/float(group['S_count'])).to('cpu')

                det_A = torch.det(A)
                det_S = torch.det(S)

                # Compute eigenvalues and eigenvectors of A and S
                la, Qa = torch.linalg.eigh(A, UPLO='U')
                ls, Qs = torch.linalg.eigh(S, UPLO='U')
                eigenval_diags = torch.outer(la, ls).t()

            G_list[group['mod']]['Qa'] = Qa.to(self.device)
            G_list[group['mod']]['Qs'] = Qs.to(self.device)
            G_list[group['mod']]['lambda'] = eigenval_diags.to(self.device)
            # For testing purposes
            G_list[group['mod']]['A'] = A
            G_list[group['mod']]['S'] = S
            G_list[group['mod']]['A_inv'] = torch.inverse(A)
            G_list[group['mod']]['S_inv'] = torch.inverse(S)

            
        return G_list
