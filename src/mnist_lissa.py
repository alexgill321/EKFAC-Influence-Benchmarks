from torch_influence import BaseObjective, LissaInfluenceModule
from torchvision import datasets, transforms
import torch
from linear_nn import get_model, load_model
from torch.utils.data import Subset

def main():
    net, _, _= get_model()
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

    class ClassObjective(BaseObjective):
        def train_outputs(self, model, batch):
            return model(batch[0])
        
        def train_loss_on_outputs(self, outputs, batch):
            criterion = torch.nn.CrossEntropyLoss()
            return criterion(outputs, batch[1])
        
        def train_regularization(self, params):
            return torch.square(params.norm())
        
        def test_loss(self, model, params, batch):
            outputs = model(batch[0])
            criterion = torch.nn.CrossEntropyLoss()
            return criterion(outputs, batch[1])
    



