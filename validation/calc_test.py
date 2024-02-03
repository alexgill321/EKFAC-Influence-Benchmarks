import unittest
import torch

class TestCalculations(unittest.TestCase):
    def setUp(self):
        pass

    def test_avg_act_matrices(self):
        group = {}
        group['A'] = []

        group['A'].append(torch.tensor([[-1, 1], [-1, 1]]).type(torch.float32))
        group['A'].append(torch.tensor([[-1, -1], [1, 1]]).type(torch.float32))
        group['A'].append(torch.tensor([[-1, 1], [-1, 1]]).type(torch.float32))

        avg_act = torch.tensor([[-1, 1/3], [-1/3, 1]]).type(torch.float32)

        A = torch.stack(group['A']).mean(dim=0)

        print(A)

        print (avg_act)
        self.assertTrue(torch.allclose(avg_act, A))

    def test_ihvp_calc(self):
        A = torch.tensor([[2, -1, 1, 1], [-1, 1, 0, 2], [1, 0, 3, 1], [1, 2, 1, 4]]).type(torch.float32)
        S = torch.tensor([[1, 2],[2, 0]]).type(torch.float32)

        la, Qa = torch.linalg.eigh(A, UPLO='U')
        ls, Qs = torch.linalg.eigh(S, UPLO='U')
        eigenval_diags = torch.outer(la, ls).flatten(start_dim=0)
        x = torch.randn(1, 4)
        gy = torch.randn(1, 2)

        x_kfe = torch.mm(x, Qa)
        gy_kfe = torch.mm(gy, Qs)
        m2 = torch.mm(gy_kfe.t()**2, x_kfe**2)

        m22 = torch.mm(Qs.t(), torch.mm(torch.mm(gy.t(), x), Qa))**2

        m3 = torch.outer(la, ls).t()

        grads = torch.mm(gy.t(), x)

        t_Qa = torch.t(Qa)
        t_Qs = torch.t(Qs)

        eps = 1e-5

        inner_num = torch.matmul(t_Qs, torch.matmul(grads, Qa))
        inv_diag = 1/(m2 + eps)
        
        inner_prod = torch.div(inner_num, inv_diag)
        outer_prod = torch.matmul(Qs, torch.matmul(inner_prod, t_Qa))
        ihvp_real = torch.flatten(outer_prod)

        training_grad = grads.flatten().t()

        if_score = torch.matmul(ihvp_real, training_grad)
        
        grad2 = torch.matmul(x.t(), gy)
        mm1 = torch.matmul(grads, t_Qa)
        mm2 = torch.matmul(Qs, mm1)
        rec = torch.reciprocal(m2 + eps)
        mm3 = torch.matmul(mm2/rec, Qa)
        ihvp = torch.matmul(t_Qs, mm3)


        print(m2)
        print(m22)
        print(m2.shape)
        print(eigenval_diags)
        print(la)
        print(Qa)
        print(ls)
        print(Qs)


