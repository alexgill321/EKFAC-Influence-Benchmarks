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
    test_dataset = Subset(train_dataset, range(500))
    
    influence_model = EKFACInfluence(model, layers=['fc1', 'fc2'], influence_src_dataset=train_dataset, batch_size=128, cov_batch_size=1)
    influences = influence_model.kfac_influence(test_dataset)

    for layer in influences:
        print(layer)
        print(influences[layer].shape)
        print(influences[layer][0].shape)
        print(torch.argmax(influences[layer][0]))
        print(torch.argmax(influences[layer][1]))
    

    k = 5
    with open('top_influences.txt', 'w') as file:
        for layer in influences:
            file.write(f'{layer}\n')
            file.write(f'Shape: {influences[layer].shape}\n')
            for i, influence in enumerate(influences[layer]):
                top = torch.topk(influence, k=k).indices
                file.write(f'Sample {i}  Top {k} Influence Indexes: {[val for val in top.tolist()]}\n')

if __name__ == '__main__':
    main()
