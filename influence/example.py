from base import BaseObjective
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import sys
sys.path.append('c:\\Users\\alexg\\Documents\\GitHub\\EKFAC-Influence-Benchmarks')

from src.linear_nn import get_model, load_model
from modules import KFACInfluenceModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNISTObjective(BaseObjective):
    def train_outputs(self, model, batch):
        return model(batch[0].to(DEVICE))

    def train_loss_on_outputs(self, outputs, batch):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs, batch[1].to(DEVICE))

    def test_loss(self, model, params, batch):
        outputs = model(batch[0].to(DEVICE))
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs, batch[1].to(DEVICE))
    
def main():
    net, _, _= get_model()
    model = load_model(net, os.getcwd() + '/models/linear_trained_model.pth')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    train_subset = Subset(train_dataset, range(2000))
    test_subset = Subset(train_dataset, range(100))

    train_idxs = list(range(0, 1000))
    test_idxs = list(range(0, 10))
    train_dataloader = DataLoader(train_subset, batch_size=32)
    test_dataloader = DataLoader(test_subset, batch_size=2)

    module = KFACInfluenceModule(
        model=model,
        objective=MNISTObjective(),
        layers=['fc1', 'fc2'],
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        device=DEVICE,
        damp=1e-4
    )

    module.influences(train_idxs, test_idxs)

if __name__ == '__main__':
    main()