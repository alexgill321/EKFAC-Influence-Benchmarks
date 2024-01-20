from EKFAC_Pytorch.kfac import KFAC
from torchvision import datasets, transforms
import torch
from linear_nn import get_model, load_model
from torch.utils.data import Subset, DataLoader, Dataset
import os
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_subset = Subset(train_dataset, range(1000))
    
    kfac_conditioner = KFAC(
        model=model, 
        eps=1e-4,
    )

    for i, (inputs, targets) in enumerate(tqdm(train_subset)):
        model.zero_grad()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        kfac_conditioner.step()
    
    state = kfac_conditioner.state_dict()
