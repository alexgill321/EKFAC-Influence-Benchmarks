import tqdm
import torch
import torchvision
import torch.nn as nn
import copy
from torch import Tensor
from torch.nn import Module

from typing import Any, Dict, List, Union, Tuple

import sys
sys.path.append('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks')

from src.linear_nn import SimpleNN, load_data, get_model, load_model

import torch

def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:

	if len(names) == 1:
		delattr(obj, names[0])
	else:
		_del_nested_attr(getattr(obj, names[0]), names[1:])

def extract_weights(mod: nn.Module):

	orig_params = tuple(mod.parameters())
	names = []
	for name, p in list(mod.named_parameters()):
		_del_nested_attr(mod, name.split("."))
		names.append(name)
	params = tuple(p.detach().requires_grad_() for p in orig_params)
	return params, names

def _set_nested_attr(obj: Module, names: List[str], value: Tensor) -> None:

	if len(names) == 1:
		setattr(obj, names[0], value)
	else:
		_set_nested_attr(getattr(obj, names[0]), names[1:], value)

def load_weights(mod: Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
	for name, p in zip(names, params):
		_set_nested_attr(mod, name.split("."), p)


def jacobian(model, x, targets):
    jac_model = copy.deepcopy(model)
    all_params, all_names = extract_weights(jac_model)
    load_weights(jac_model, all_names, all_params)

    def param_as_input_func(model, x, param):
        load_weights(model, [name], [param])
        out = model(x)
        loss = criterion(out, targets)
        return loss

    jac = ''
    param_weight = ''

    for i, (name, param) in enumerate(zip(all_names, all_params)):
        jac = torch.autograd.functional.jacobian(lambda param: param_as_input_func(jac_model, x, param), param,
                             strict=True if i==0 else False, vectorize=False if i==0 else True)

        param_weight = param
        print("for the param name : {} the shape was : {} and the jacobian shape is {}".format(name, str(i), str(jac.shape)))
        break

    del jac_model # cleaning up
    return jac.squeeze(dim =0), param_weight


def init_all_objects():

    print("got the data.")
    model, criterion, optimizer = get_model()
    trained_model = load_model(model, filepath='models/linear_trained_model.pth')
    print("got the model.")
    train_loader, val_loader, test_loader = load_data()

    return trained_model, train_loader, val_loader, criterion

def compute_pbrf(model, gnh_matrix):
    pbrf_params = ''
    return pbrf_params

def compute_influence_and_top_examples(model, pbrf_params, test_query):
    highest_loss_difference_example = ''
    influence_score_from_pbrf=''
    return influence_score_from_pbrf

if __name__ == '__main__':
    trained_model, train_loader, val_loader, criterion = init_all_objects()

    x = torch.ones(4, requires_grad=True)
    inputs = ''
    targets = ''
    loss_on_current_example = 0

    itr = 0
    for input, targets in train_loader:

        inputs = torch.tensor(input, requires_grad=True)
        targets = targets

        outputs = trained_model(inputs)
        loss_on_current_example = criterion(outputs, targets)

        if itr>10:
            break

    jacobain_matrix, param_weight = jacobian(trained_model, inputs, targets)
    print("jacobian shape is {}".format(jacobain_matrix.shape))

    hessian_approximation = jacobain_matrix.T.matmul(jacobain_matrix)

    print(hessian_approximation.shape)
    jacob_hessian = torch.matmul(jacobain_matrix, hessian_approximation)
    print("jacob hedssian is of shape")
    print(jacob_hessian.shape)
    final_prod = torch.matmul(jacob_hessian, jacobain_matrix.t())
    print(final_prod.shape)

    identity_tensor = torch.eye(final_prod.shape[0], final_prod.shape[1])
    print(identity_tensor.shape)
    lamdba_eye = 0.01 * identity_tensor
    final_gnh_scaled = final_prod + lamdba_eye
    print("gnh shape is ")
    print(final_gnh_scaled.shape)
    gnh_inverse = torch.pinverse(final_gnh_scaled)

    print("gnh inverse shape is {}".format(gnh_inverse.shape))

    grad_of_loss_wrt_params = torch.autograd.grad(loss_on_current_example, trained_model.fc1.weight, create_graph=True)
    print("grad of loss of pred wrt params shape is {}".format(grad_of_loss_wrt_params[0].shape))
    print("params shape is : {}".format(trained_model.fc1.weight.shape))

    final_pbrf_for_current_example = trained_model.fc1.weight + torch.matmul(gnh_inverse,
                                                                             grad_of_loss_wrt_params[0])
    print("final pbrf score")
    print(final_pbrf_for_current_example.shape)


