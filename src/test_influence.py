import sys

from linear_nn import get_model, load_model
from model_eval import train_dataset
from influence_utils import EKFACInfluence
import torch
import os


def main():
    net, _, _ = get_model()
    model = load_model(net, os.getcwd() + '/models/linear_trained_model.pth')
    
    influence_model = EKFACInfluence(model, layers=['fc1', 'fc2'], influence_src_dataset=train_dataset, batch_size=128, cov_batch_size=128)

