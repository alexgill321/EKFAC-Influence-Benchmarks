import time

import numpy as np
import torch
import torchvision
import deepwave


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float


ny = 500
nx = 256
freq = 25
nt = 80
dt = 0.004
dx = 4

v_true = 1500 * torch.ones(ny, nx, dtype=dtype, device=device)
v_true[10:] += 200
v_init = (torchvision.transforms.functional.gaussian_blur(
    v_true[None], [31, 31]).squeeze())
v = v_init.clone().requires_grad_()


print(v.shape)


source_locations = torch.tensor([[[1, 15]]], dtype=torch.long, device=device)
source_amplitudes = deepwave.wavelets.ricker(freq,
                                             nt,
                                             dt,
                                             1.3 / freq,
                                             dtype=dtype).reshape(1, 1, -1)
receiver_locations = torch.ones(1, nx - 20, 2, dtype=torch.long, device=device)
receiver_locations[0, :, 1] = torch.arange(10, nx - 10)


d_true = deepwave.scalar(
    v_true,
    grid_spacing=dx,
    dt=dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_width=[0, 20, 20, 20],
    max_vel=2000,
)[-1]

loss_fn = torch.nn.MSELoss()
def wrap(v):
    # d = deepwave.scalar(
    #     v,
    #     grid_spacing=dx,
    #     dt=dt,
    #     source_amplitudes=source_amplitudes,
    #     source_locations=source_locations,
    #     receiver_locations=receiver_locations,
    #     pml_width=[0, 20, 20, 20],
    #     max_vel=2000,
    # )[-1]
    random_loss_vector = torch.rand(1)
    return random_loss_vector
    # return loss_fn(d, d_true)

tt = time.time()
print("starting the hessian calc")
hess = torch.autograd.functional.hessian(wrap, v, vectorize = True).detach()
print("total operation time is : {}".format(time.time()- tt))

print(hess.shape)