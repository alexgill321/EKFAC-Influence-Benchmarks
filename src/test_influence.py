import sys

from linear_nn import get_model, load_model
from influence_utils import EKFACInfluence
from torchvision import datasets, transforms
import torch
import os
from torch.utils.data import Subset


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
    test_dataset = Subset(train_dataset, range(500))
    
    influence_model = EKFACInfluence(model, layers=['fc1', 'fc2'], influence_src_dataset=train_dataset, batch_size=128, cov_batch_size=1)
    influences = influence_model.influence(test_dataset)

    for layer in influences:
        print(layer)
        print(influences[layer].shape)
        print(influences[layer][0].shape)
        print(torch.argmax(influences[layer][0]))
        print(torch.argmax(influences[layer][1]))
    
    for layer in influences:
        test_influences = influences[layer].detach().clone()
        for i, influence in enumerate(test_influences):
            print(influence[:10])
            print(torch.max(influence))
            top = torch.argmax(influence)
            influence[top] = 0
            count = 0
            while top != i:
                influence[top] = 0
                count += 1
                top = torch.argmax(influence)
            print(f"top influence found in {count} steps")
        break

if __name__ == '__main__':
    main()
