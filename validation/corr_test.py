import os
from torchvision import datasets, transforms
import torch
import sys
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterSampler


# sys.path.append('/Users/purbidbambroo/PycharmProjects/EKFAC-Influence-Benchmarks/')
sys.path.append('/Users/alexg/Documents/GitHub/EKFAC-Influence-Benchmarks')

from torchinfluenceoriginal.torch_influence.base import BaseObjective
from torchinfluenceoriginal.torch_influence.modules import  LiSSAInfluenceModule


from tqdm import tqdm

from src.linear_nn import get_model, load_model
from influence.modules import EKFACInfluenceModule, IHVPInfluence
from influence.base import BaseInfluenceObjective

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L2_WEIGHT = 1e-4


def main():
    net, criterion, _= get_model()
    model = load_model(net, os.getcwd() + '/models/linear_trained_model.pth')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    generator = torch.Generator().manual_seed(69)

    train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

    random_train = torch.randperm(len(train_dataset), generator=generator)[:500]
    random_test = torch.randperm(len(test_dataset), generator=generator)[:10]

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    # if not os.path.exists(os.getcwd() + '/results/lissa_influences.txt'):
    #     generate_lissa_influences(model, train_dataloader, test_dataloader, random_train, random_test)

    # if not os.path.exists(os.getcwd() + '/results/ekfac_refactored_influences_fc1.txt'):
    #generate_ekfac_refac_influences(model, train_dataloader, test_dataloader, random_train, random_test)

    # if not os.path.exists(os.getcwd() + '/results/PBRF_influence_scores_random.txt'):
    generate_pbrf_refac_influences(model, train_dataloader, test_dataloader, random_train, random_test, criterion)


def generate_lissa_influences(model, train_dataloader, test_dataloader, random_train, random_test):
    # train_dataset_sub = CustomSubsetDataset(train_dataset, random_train)
    # test_dataset_sub = CustomSubsetDataset(test_dataset, random_test)

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
        
    lissa_module = LiSSAInfluenceModule(
        model = model,
        objective = ClassObjective(),
        train_loader = train_dataloader,
        test_loader = test_dataloader,
        device = DEVICE,
        damp = 1e-5,
        repeat = 1,
        depth = (int) (len(train_dataloader.dataset)/train_dataloader.batch_size),
        scale = 100,
        gnh = True,
    )

    influences = []
    for test_idx in tqdm(random_test, desc='Calcualting Influence'):
        influences.append(lissa_module.influences(random_train, [test_idx]))
    
    if not os.path.exists(os.getcwd() + '/results'):
        os.mkdir(os.getcwd() + '/results')

    with open(os.getcwd() + '/results/lissa_influences.txt', 'w') as file:
        for test_idx, influence in zip(random_test, influences):
            file.write(f'{test_idx}: {influence.tolist()}\n')

def generate_ekfac_refac_influences(model, train_dataloader, test_dataloader, random_train, random_test):
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

    for damp_value in [0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        print("for damp factor {}".format(damp_value))
        module = EKFACInfluenceModule(
            model=model,
            objective=MNISTObjective(),
            layers=['fc1', 'fc2'],
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            device=DEVICE,
            damp=damp_value,
            n_samples=2
        )

        influences = module.influences(random_train, random_test)

        if not os.path.exists(os.getcwd() + '/results'):
            os.mkdir(os.getcwd() + '/results')

        for layer in influences:
            with open(os.getcwd() + f'/results/ekfac_refactored_influences_{layer}_scaling_{damp_value}.txt', 'w') as file:
                for test_idx, influence in zip(random_test, influences[layer]):
                    file.write(f'{test_idx}: {influence.tolist()}\n')
            file.close()


def generate_pbrf_refac_influences(model, train_dataloader, test_dataloader, random_train, random_test, criterion):
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
        
    for damp in [.01, .1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:

    # for damp_criterion in [0.1]:
        # , 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        module = IHVPInfluence(
            model=model,
            objective=MNISTObjective(),
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            device=DEVICE,
            damp=damp,
            criterion=criterion,
        )

        influences = []

        for test_idx in tqdm(random_test, desc='Computing Influences'):
            influences.append(module.influences(random_train, [test_idx], ihvps_for="pbrf"))

        if not os.path.exists(os.getcwd() + '/results'):
            os.mkdir(os.getcwd() + '/results')

        with open(os.getcwd() + '/results/PBRF_influence_scores_random_scaling_{}_epsilon_{}.txt'.format(param['scaling_factor'], param['downweight_factor']), 'w') as file:

            for test_idx, influence in zip(random_test, influences):
                file.write(f'{test_idx}: {torch.mul(influence, 1).tolist()}\n')
        file.close()


if __name__ == '__main__':
    main()

