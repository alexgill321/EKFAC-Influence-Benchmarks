from torch_influence import BaseObjective, LiSSAInfluenceModule, CGInfluenceModule
from torchvision import datasets, transforms
import torch
from linear_nn import get_model, load_model
from torch.utils.data import Subset, DataLoader
import os
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L2_WEIGHT = 1e-4

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
    test_dataset = Subset(train_dataset, range(10))

    class ClassObjective(BaseObjective):
        def train_outputs(self, model, batch):
            return model(batch[0])
        
        def train_loss_on_outputs(self, outputs, batch):
            criterion = torch.nn.CrossEntropyLoss()
            return criterion(outputs, batch[1])
        
        def train_regularization(self, params):
            return L2_WEIGHT * torch.square(params.norm())
        
        def test_loss(self, model, params, batch):
            outputs = model(batch[0])
            criterion = torch.nn.CrossEntropyLoss()
            return criterion(outputs, batch[1])
        
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
        
    module = LiSSAInfluenceModule(
        model = model,
        objective = ClassObjective(),
        train_loader = train_dataloader,
        test_loader = test_dataloader,
        device=DEVICE,
        damp=1e-2,
        repeat=10,
        depth=50,
        scale=.995,
        gnh=True
    )

    cg_module = CGInfluenceModule(
        model = model,
        objective = ClassObjective(),
        train_loader = train_dataloader,
        test_loader = test_dataloader,
        device=DEVICE,
        damp=1e-4,
        gnh=False
    )
    
    train_idxs = list(train_dataset.indices)
    test_idxs = list(test_dataset.indices)

    for test_idx in tqdm(test_idxs, desc="Computing Influences"):
        influences = cg_module.influences(train_idxs=train_idxs, test_idxs=[test_idx])
        print(influences.shape)

if __name__ == '__main__':
    main()

