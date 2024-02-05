#%%
from matplotlib import patches
from torchvision import datasets
from torch.utils.data import Subset
import os
import ast
import matplotlib.pyplot as plt
import torch


def plot_top_influences(inf_src, n, k=5, label=None):

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
                    lst = ast.literal_eval(splits[1])
                    array_list.append(lst)
                except ValueError as e:
                    continue
            else:
                continue

    for j, lst in enumerate(array_list):
        fig, axes = plt.subplots(1, 6, figsize=(15, 3)) 
        # Iterate over image paths and axes to plot each image
        for i, (index, ax) in enumerate(zip(lst, axes[1:k+1])):
            image = train_dataset[index][0]  # Load the image
            # Display the image on the current axis
            ax.imshow(image)
            ax.axis('off')  # Turn off axis labels
            ax.set_title(f"Influence #{i+1}: IMG {index}")
        
        image = train_dataset[j][0]
        axes[0].imshow(image)
        
        # Add a border to axes[0]
        axes[0].add_patch(patches.Rectangle((0, 0), 27, 27, linewidth=2, edgecolor='r', facecolor='none'))
        
        axes[0].axis('off')
        axes[0].set_title(f"Test Image", fontweight='bold')

        if label:
            plt.suptitle(f"Top {k} Influences for Image {j}: {label}")
        else:
            plt.suptitle(f"Top {k} Influences for Image {j}")

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
            parts = line.strip().split(':')
            influence = ast.literal_eval(parts[1])
            influences_1.append(influence)
    
    influences_2 = []
    with open(inf_src2, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(':')
            influence = ast.literal_eval(parts[1])
            influences_2.append(influence)

    corr_list = []
    for i, j in zip(influences_1, influences_2):
        inf_1 = torch.tensor(i)
        inf_2 = torch.tensor(j)
        inf_stacked = torch.stack([inf_1, inf_2], dim=0)
        corr_list.append(torch.corrcoef(inf_stacked)[0, 1].item())
    print(f'Average Correlation for {inf_src1} and {inf_src2}: {sum(corr_list) / len(corr_list)}')

    

if __name__ == '__main__':
    # Replace with the path to your top_influences.txt file
    # plot_top_influences(os.getcwd() + '/results/top_influences.txt', 5)
    # plot_top_influences(os.getcwd() + '/results/top_influences_lissa.txt', 5)
    # lissa_influences = os.getcwd() + '/results/lissa_influences.txt'
    ekfac_refac_influences = os.getcwd() + '/results/ekfac_refactored_influences_fc2.txt'
    pbrf_influences = os.getcwd() + '/results/pbrf_influences_fc2.txt'
    # influence_correlation(ekfac_influences, refac_ekfac_influences)
    # influence_correlation(lissa_influences, ekfac_influences)
    # influence_correlation(lissa_influences, ekfac_refac_influences)
    # influence_correlation(lissa_influences, pbrf_influences)
    influence_correlation(ekfac_refac_influences, pbrf_influences)
    # plot_top_influences(os.getcwd() + '/results/refac_ekfac_top_influences.txt', 10, label='REKFAC')
