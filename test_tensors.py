import torch

good = torch.load('good_grad.pt')

bad = torch.load('bad_grad.pt')

print(torch.equal(good, bad))