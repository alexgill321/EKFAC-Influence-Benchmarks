#%%
import argparse
import itertools
from matplotlib import patches
import pandas as pd
from torchvision import datasets
from torch.utils.data import Subset
import os
import ast
import matplotlib.pyplot as plt
import torch
from IPython.display import display


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
    return sum(corr_list) / len(corr_list)

    

if __name__ == '__main__':

    lissa_influences = os.getcwd() + '/results/lissa_influences.txt'

    print("getting all PBRF scores")

# pbrf_influences = os.getcwd() + '/results/PBRF_influence_scores_random_scaling_0.001_epsilon_120000.txt'

    pbrf_files = [filename for filename in os.listdir(os.getcwd() + '/results/') if filename.startswith('PBRF_influence_scores') or filename.startswith('pbrf_influences_fc2_')]

    print("getting all EKFAC scores")

    ekfac_files = [filename for filename in os.listdir(os.getcwd() + '/results/') if filename.startswith('ekfac_influences_fc2_')]


    pairs = list(itertools.product(ekfac_files, pbrf_files))
    
    all_corr_data = []
    for  ekfac_file, pbrf_file in pairs:
        try:
            ekfac_damp = ekfac_file.split("_")[-1][:-4]
            pbrf_damp = pbrf_file.split("_")[-1][:-4]

            corr = influence_correlation(os.getcwd() + '/results/' + ekfac_file, os.getcwd() + '/results/' + pbrf_file)
            all_corr_data.append([ekfac_damp, pbrf_damp, corr])
        except Exception as e:
            print(e)

        column_names = ['ekfac_damp', 'pbrf_damp', 'correlation']

        df_hyper = pd.DataFrame(data = all_corr_data, columns=column_names)

        df_hyper.to_csv(os.getcwd() + '/results/damp_search_results.csv', index=False)
    
    display(df_hyper)