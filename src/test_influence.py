import sys

from linear_nn import get_model, load_model
from influence_utils import EKFACInfluence
from torchvision import datasets, transforms
import torch
import os
from torch.utils.data import Subset
import matplotlib.pyplot as plt

def main():
    net, _, _ = get_model()
    model = load_model(net, os.getcwd() + '/models/linear_trained_model.pth')
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)), 
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    # Download MNIST dataset and create DataLoader
    train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    train_dataset = Subset(train_dataset, range(1000))
    test_dataset = Subset(train_dataset, range(10))
    
    influence_model = EKFACInfluence(model, layers=['fc1', 'fc2'], influence_src_dataset=train_dataset, batch_size=1, cov_batch_size=1)
    influences = influence_model.influence(test_dataset, eps=1e-5)
    
    if not os.path.exists(os.getcwd() + '/results'):
        os.mkdir(os.getcwd() + '/results')

    k = 10
    with open(os.getcwd() + '/results/ekfac_top_influences.txt', 'w') as file:
        for layer in influences:
            file.write(f'{layer}\n')
            for i, influence in enumerate(influences[layer]):
                top = torch.topk(influence, k=k).indices
                file.write(f'Sample {i}  Top {k} Influence Indexes: {[val for val in top.tolist()]}\n')

    for layer in influences:
        with open(os.getcwd() + f'/results/ekfac_influences_{layer}.txt', 'w') as file:
            for i, influence in enumerate(influences[layer]):
                file.write(f'{i}: {influence.tolist()}\n')
        file.close()

if __name__ == '__main__':
    main()
