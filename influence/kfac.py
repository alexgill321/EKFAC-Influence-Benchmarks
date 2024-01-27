from torch.optim.optimizer import Optimizer
import torch


class KFAC(Optimizer):
    def __init__(self, net):
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self.net = net
        self.calc_act = True

        # Register hooks for activations and gradients
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
                    
                # Defining the parameter dictionary to store per layer calculations
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)
        super(KFAC, self).__init__(self.params, {})

    def step(self):
        """Performs a single optimization step.
        
        This method is called once per optimizer step. It computes the covariance matrices
        for the layer activations and the layer output gradients.
        """

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

            self._calc_cov(group, x, gy)

    def kfac_diag(self, G_list, kfac_diags):
        for group in self.param_groups:
            mod = group['mod']

            Qa = G_list[mod]['Qa']
            Qs = G_list[mod]['Qs']

            x = self.state[mod]['x']
            gy = self.state[mod]['gy']

            if mod.bias is not None:
                x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

            gy_kfe = torch.mm(gy, Qs)
            x_kfe = torch.mm(x, Qa)
            kfac_diag = (torch.mm(gy_kfe.t() ** 2, x_kfe ** 2).view(-1))

            if mod not in kfac_diags:
                kfac_diags[mod] = kfac_diag
            else:
                kfac_diags[mod].add_(kfac_diag)

        return kfac_diags
            
    def calc_A(self, group, x):
        """ Calculates and updates the value of 'A' in the given group.

        Args:
            group (dict): The dictionary containing the group of parameters.
            x (tensor): The input tensor.
        """
        if self.calc_act:
            # Calculate covariance matrix for layer activations (A_{l})
            if 'A' not in group:
                group['A'] = torch.matmul(x, x.t())/float(x.shape[1])
                group['A_count'] = 1
            else:
                torch.add(group['A'], torch.matmul(x, x.t())/float(x.shape[1]), out=group['A'])
                group['A_count'] += 1
            
        
    def calc_S(self, group, gy):
            """ Calculates and updates the value of 'S' in the given group.

            Args:
                group (dict): The group dictionary.
                gy (torch.Tensor): The input tensor.
            """
            if 'S' not in group:
                group['S'] = torch.matmul(gy, gy.t())/float(gy.shape[1])
                group['S_count'] = 1
            else:
                torch.add(group['S'], torch.matmul(gy, gy.t())/float(gy.shape[1]), out=group['S'])
                group['S_count'] += 1