from base import BaseInfluenceObjective
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import os
import torch
import sys  

sys.path.append('c:\\Users\\alexg\\Documents\\GitHub\\EKFAC-Influence-Benchmarks')
from src.linear_nn import get_model, load_model
from modules import PBRFInfluenceModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNISTObjective(BaseInfluenceObjective):
    def train_outputs(self, model, batch):
        return model(batch[0].to(DEVICE))
    
    def train_loss_on_outputs(self, outputs, batch):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, batch[1].to(DEVICE))
        return loss
    
    def test_loss(self, model, batch):
        outputs = model(batch[0].to(DEVICE))
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs, batch[1].to(DEVICE))
    
def main():
    net, _, _= get_model()
    model = load_model(net, os.getcwd() + '/models/linear_trained_model.pth')

    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    train_subset = Subset(train_dataset, range(1000))
    test_subset = Subset(train_dataset, range(100))

    train_idxs = list(range(0, 1000))
    test_idxs = list(range(0, 100))
    train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_subset, batch_size=2, shuffle=False)

    module = PBRFInfluenceModule(
        model=model,
        objective=MNISTObjective(),
        layers=['fc2'],
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        device=DEVICE,
        damp=1e-5,
        check_eigvals=True
    )

    influences = module.influences(train_idxs, test_idxs)

    for layer in influences:
        with open(os.getcwd() + f'/results/pbrf_influences_{layer}.txt', 'w') as f:
            for i, influence in enumerate(influences[layer]):
                f.write(f'{i}: {influence.tolist()}\n')
        f.close()

    k = 10
    with open(os.getcwd() + '/results/pbrf_top_influences.txt', 'w') as file:
        for layer in influences:
            file.write(f'{layer}\n')
            for i, influence in enumerate(influences[layer]):
                top = torch.topk(influence, k=k).indices
                file.write(f'Sample {i}  Top {k} Influence Indexes: {[val for val in top.tolist()]}\n')

if __name__ == '__main__':
    main()
