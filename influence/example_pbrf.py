import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import sys

from tqdm import tqdm

sys.path.append('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/')
from influence.base import BasePBRFInfluenceModule, BaseInfluenceObjective


from src.linear_nn import get_model, load_model
from modules import EKFACInfluenceModule, IHVPInfluence

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
    net, criterion, _ = get_model()
    model = load_model(net, os.getcwd() + '/models/linear_trained_model.pth')

    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    train_subset = Subset(train_dataset, range(5000))
    test_subset = Subset(train_dataset, range(100))

    train_idxs = list(range(0, 500))
    test_idxs = list(range(0, 10))
    train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_subset, batch_size=2, shuffle=False)

    module = IHVPInfluence(
        model=model,
        objective=MNISTObjective(),
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        device=DEVICE,
        damp=1e-3,
        criterion=criterion
    )

    influences = []

    for test_idx in tqdm(test_idxs, desc='Computing Influences'):
        influences.append(module.influences(train_idxs, [test_idx], ihvps_for="pbrf"))


    if not os.path.exists(os.getcwd() + '/results'):
        os.mkdir(os.getcwd() + '/results')

    if not os.path.exists(os.getcwd() + '/results'):
        os.mkdir(os.getcwd() + '/results')

    k = 10

    with open(os.getcwd() + '/results/top_influences_PBRF_IHVP.txt', 'w') as file:
        for test_idx, influence in zip(test_idxs, influences):
            top = torch.topk(influence, k=k).indices
            file.write(f'Sample {test_idx}  Top {k} Influence Indexes: {[val for val in top.tolist()]}\n')

    with open(os.getcwd() + '/results/PBRF_influence_scores.txt', 'w') as file:

        for test_idx, influence in zip(test_idxs, influences):
            file.write(f'{test_idx}: {torch.mul(influence, 1).tolist()}\n')

if __name__ == '__main__':
    main()