import sys
# sys.path.append('/torchinfluence/')

from torchvision import datasets, transforms
import torch
from torch.utils.data import Subset, DataLoader, Dataset
import os
from tqdm import tqdm

import sys
sys.path.append('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks')
sys.path.append('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/torchinfluence/')






from src.linear_nn import get_model, load_model
from torchinfluence.torch_influence.base import BaseObjective
from torchinfluence.torch_influence.modules import IHVPInfluence, AutogradInfluenceModule


DEVICE = "cpu"
L2_WEIGHT = 1e-4


def main():
    net, criterion, _ = get_model()
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

    train_dataset_sub = CustomSubsetDataset(train_dataset, list(range(0, 500)))
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

    module = IHVPInfluence(
        model=model,
        objective=ClassObjective(),
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        device=DEVICE,
        damp=0.9,
        criterion = criterion
    )

    train_idxs = list(range(500))
    test_idxs = list(range(10))
    influences = []
    ihvps = []
    for test_idx in tqdm(test_idxs, desc='Computing Influences'):
        #this line with give the inverse hessian
        influences.append(module.influences(train_idxs, [test_idx]))
        # ihvp = module.stest([test_idx])
        # ihvp_reshaped = module._reshape_like_params(ihvp)
        # print(ihvp_reshaped[0].shape)
        # ihvp_l1 = torch.cat((ihvp_reshaped[0], ihvp_reshaped[1].reshape(-1, 1)), dim=1)
        # ihvps.append(ihvp_l1.flatten())
        # ihvps.append(ihvp_reshaped[0].flatten())


    if not os.path.exists(os.getcwd() + '/results'):
        os.mkdir(os.getcwd() + '/results')

    k = 5


    with open(os.getcwd() + '/results/top_influences_PBRF_IHVP.txt', 'w') as file:
        for test_idx, influence in zip(test_idxs, influences):
            top = torch.topk(influence, k=k).indices
            file.write(f'Sample {test_idx}  Top {k} Influence Indexes: {[val for val in top.tolist()]}\n')

    # with open(os.getcwd() + '/results/pbrf_only_ihvps.txt', 'w') as file:
    #     for test_idx, ihvp in zip(test_idxs, ihvps):
    #         file.write(f'{test_idx}: {ihvp.tolist()}\n')


    with open(os.getcwd() + '/results/PBRF_influence_scores.txt', 'w') as file:

        for test_idx, influence in zip(test_idxs, influences):
            file.write(f'{test_idx}: {torch.mul(influence, 1).tolist()}\n')
    #
    # with open(os.getcwd() + '/results/PBRF_influence_scores.txt', 'w') as file:
    #     for test_idx, ihvp in zip(test_idxs, ihvps):
    #         file.write(f'{test_idx}: {ihvp.tolist()}\n')


if __name__ == '__main__':
    main()

