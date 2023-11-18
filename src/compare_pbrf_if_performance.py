import numpy
import torch


if_tensor = torch.load('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/src/linear_layer_if_tensors.pt')

pbrf_tensors = torch.load('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/pbrf_tensor.pt')

top_result_if_tensor_score = torch.argmax(if_tensor, dim=1)






for i in range(10):
    print(pbrf_tensors[i].item())
    difference = abs(top_result_if_tensor_score[i].item() -  pbrf_tensors[i].item())
    print(difference)
    print("\n\n\n")


# print(pbrf_tensors[:480])





