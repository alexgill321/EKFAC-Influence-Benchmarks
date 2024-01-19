from torch_influence import BaseObjective, LiSSAInfluenceModule, CGInfluenceModule
from torchvision import datasets, transforms
import torch
from linear_nn import get_model, load_model
from torch.utils.data import Subset, DataLoader, Dataset
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

    class CustomSubsetDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
        
        def __len__(self):
            return len(self.indices)
        
    train_dataset_sub = CustomSubsetDataset(train_dataset, list(range(0, 1000)))
    test_dataset_sub = CustomSubsetDataset(train_dataset, list(range(0, 10)))

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
        
    train_dataloader = DataLoader(train_dataset_sub, batch_size=2)
    test_dataloader = DataLoader(test_dataset_sub, batch_size=2)
        
    module = LiSSAInfluenceModule(
        model = model,
        objective = ClassObjective(),
        train_loader = train_dataloader,
        test_loader = test_dataloader,
        device=DEVICE,
        damp=1e-4,
        repeat=10,
        depth=1000,
        scale=100,
        gnh=True
    )

    cg_module = CGInfluenceModule(
        model = model,
        objective = ClassObjective(),
        train_loader = train_dataloader,
        test_loader = test_dataloader,
        device=DEVICE,
        damp=1e-4,
        gnh=True
    )
    
    train_idxs = list(range(1000))
    test_idxs = list(range(10))
    
    influences = []
    for test_idx in tqdm(test_idxs, desc='Computing Influences'):
        influences.append(module.influences(train_idxs, [test_idx]))

    if not os.path.exists(os.getcwd() + '/results'):
        os.mkdir(os.getcwd() + '/results')

    k = 5
    with open(os.getcwd() + '/results/top_influences_lissa.txt', 'w') as file:
        for test_idx, influence in zip(test_idxs, influences):
            top = torch.topk(influence, k=k).indices
            file.write(f'Sample {test_idx}  Top {k} Influence Indexes: {[val for val in top.tolist()]}\n')
    
    with open(os.getcwd() + '/results/lissa_influences.txt', 'w') as file:
        for test_idx, influence in zip(test_idxs, influences):
            file.write(f'{test_idx}: {influence.tolist()}\n')

if __name__ == '__main__':
    main()

