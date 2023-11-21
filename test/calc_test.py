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


