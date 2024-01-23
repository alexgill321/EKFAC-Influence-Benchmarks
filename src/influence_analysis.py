#%%
from torchvision import datasets
from torch.utils.data import Subset
import os
import ast
import matplotlib.pyplot as plt
import torch


def plot_top_influences(inf_src, n, k=5):

    train_dataset = datasets.MNIST(root='../data', train=True, download=True)
    
    array_list = []

    # Replace with the path to your top_influences.txt file
    with open(inf_src, 'r') as file:
        for line in file:
            if line.startswith('S'):
                line = line.strip()
                splits = line.split(': ')
                if len(splits) != 2:
                    continue
                try:
                    list = ast.literal_eval(splits[1])
                    array_list.append(list)
                except ValueError as e:
                    continue
            else:
                continue

    for j, list in enumerate(array_list):
        fig, axes = plt.subplots(1, 6, figsize=(15, 3)) 
        # Iterate over image paths and axes to plot each image
        for i, (index, ax) in enumerate(zip(list, axes[:k])):
            image = train_dataset[index][0]  # Load the image
            # Display the image on the current axis
            ax.imshow(image)
            ax.axis('off')  # Turn off axis labels
            ax.set_title(f"Image {index}")
        
        image = train_dataset[j][0]
        axes[5].imshow(image)
        axes[5].axis('off')
        axes[5].set_title(f"Actual Image {j}")

        plt.suptitle(f"Top {k} Influences for Image {j}: {inf_src}")

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

        if j == n:
            break

def influence_correlation(inf_src1, inf_src2):
    influences_1 = []
    with open(inf_src1, 'r') as file:
        lines = file.readlines()
        for line in lines:
            influence = ast.literal_eval(line[3:])
            influences_1.append(influence)
    
    influences_2 = []
    with open(inf_src2, 'r') as file:
        lines = file.readlines()
        for line in lines:
            influence = ast.literal_eval(line[3:])
            influences_2.append(influence)

    corr_list = []
    for i, j in zip(influences_1, influences_2):
        inf_1 = torch.tensor(i)
        inf_2 = torch.tensor(j)
        inf_stacked = torch.stack([inf_1, inf_2], dim=0)
        corr_list.append(torch.corrcoef(inf_stacked)[0, 1].item())
    print(f'Average Correlation: {sum(corr_list) / len(corr_list)}')

    

if __name__ == '__main__':
    # Replace with the path to your top_influences.txt file
    # plot_top_influences(os.getcwd() + '/results/top_influences.txt', 5)
    # plot_top_influences(os.getcwd() + '/results/top_influences_lissa.txt', 5)
    influence_correlation(os.getcwd() + '/results/lissa_influences.txt', os.getcwd() + '/results/kfac_influences_Linear(in_features=784, out_features=256, bias=True).txt')
    plot_top_influences(os.getcwd() + '/results/kfac_top_influences.txt', 10)
