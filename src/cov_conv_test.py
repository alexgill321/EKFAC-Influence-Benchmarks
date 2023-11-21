from linear_nn import get_model, load_model
from influence_utils import EKFACInfluence
from torch.utils.data import random_split, Subset
import captum._utils.common as common
from torchvision import datasets, transforms
import os
import torch
import matplotlib.pyplot as plt

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
    cov_dataset_1 = Subset(train_dataset, range(int(len(train_dataset)/2), len(train_dataset)))
    influence_model = EKFACInfluence(model, layers=['fc1', 'fc2'], influence_src_dataset=cov_dataset_1, batch_size=128, cov_batch_size=1)
    G_list = influence_model._compute_EKFAC_params()

    # influence_model2 = EKFACInfluence(
    #     model, 
    #     layers=['fc1', 'fc2'], 
    #     influence_src_dataset=cov_dataset_2, 
    #     batch_size=128, 
    #     cov_batch_size=128
    #     )
    # G_list2 = influence_model2._compute_EKFAC_params()

    layer_modules = [common._get_module_from_name(model, layer) for layer in influence_model.layers]

    # for layer in layer_modules:
    #     print(f'Cov matrices for layer {layer}')
    #     dif_A = G_list[layer]["A"] - G_list2[layer]["A"]
    #     dif_S = G_list[layer]["S"] - G_list2[layer]["S"]
    #     print(f'First row of dif_A: {dif_A[0]}')
    #     print(f'First row of dif_S: {dif_S[0]}')
    #     print(f'Mean difference in activations cov: {torch.mean(torch.abs(dif_A)).item()}')
    #     print(f'Mean difference in pseudograd cov: {torch.mean(torch.abs(dif_S)).item()}') 

    # Is it converging?
    print('Convergence test')
    mean_diff = {}
    for layer in layer_modules:
        mean_diff[layer] = {}
        mean_diff[layer]["A"] = []
        mean_diff[layer]["S"] = []
    for i in range(3, 100, 3):
        print(i)
        subset_dataset = Subset(train_dataset, range(i))
        influence_model2 = EKFACInfluence(
            model, 
            layers=['fc1', 'fc2'], 
            influence_src_dataset=subset_dataset, 
            batch_size=128, 
            cov_batch_size=1
            )
        G_list2 = influence_model2._compute_EKFAC_params()
        for layer in layer_modules:
            dif_A = G_list[layer]["A"] - G_list2[layer]["A"]
            dif_S = G_list[layer]["S"] - G_list2[layer]["S"]
            mean_diff[layer]["A"].append(torch.mean(torch.abs(dif_A)).item())
            mean_diff[layer]["S"].append(torch.mean(torch.abs(dif_S)).item())
            print(f'Mean difference in activations cov: {torch.mean(torch.abs(dif_A)).item()}')
            print(f'Mean difference in pseudograd cov: {torch.mean(torch.abs(dif_S)).item()}')

    plt.figure()
    for layer in layer_modules:
        plt.plot(mean_diff[layer]["A"], label=f'{layer} activations')
    
    plt.legend()
    plt.show()

    plt.figure()
    for layer in layer_modules:
        plt.plot(mean_diff[layer]["S"], label=f'{layer} pseudograd')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()



