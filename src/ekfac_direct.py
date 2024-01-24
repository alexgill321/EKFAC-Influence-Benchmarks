import os
from torchvision import datasets, transforms
from ekfac import EKFAC
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
    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)

    tester = EKFACTester(model, 0.0001, train_loader)
    tester.run()

class EKFACTester():
    def __init__(self, model, eps, train_loader):
        self.model = model
        self.ekfac = EKFAC(
            self.model,
            eps=eps,
            alpha=1.0
        )
        self.train_loader = train_loader
    
    def run(self):
        for i, (inputs, targets) in enumerate(tqdm(self.train_loader)):
            self.model.train()
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = self.model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            self.ekfac.step(update_params=False)
            self.model.zero_grad()

        state = self.ekfac.state_dict()
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.Linear) or isinstance(mod, torch.nn.Conv2d):
                res = state['state'][mod]
                print(res)

if __name__ == '__main__':
    main()