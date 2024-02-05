#%%
import argparse

import pandas as pd
from matplotlib import patches
from torchvision import datasets
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
    return sum(corr_list) / len(corr_list)

    

if __name__ == '__main__':
    # Replace with the path to your top_influences.txt file

    parser = argparse.ArgumentParser("N-gram Language Model")
    parser.add_argument('--do_PBRF_search', type=bool, required=False, default=False)
    parser.add_argument('--do_EKFAC_search', type=bool, required=False, default=False)
    parser.add_argument('--best_PBRF_path', type=str, required=False, default='/results/PBRF_influence_scores_random_scaling_0.001_epsilon_120000.txt')
    parser.add_argument('--best_EKFAC_path', type=str, required=False, default='/results/ekfac_refactored_influences_fc2_scaling_0.01.txt')

    args = parser.parse_args()

    args.do_EKFAC_search = True


    lissa_influences = os.getcwd() + '/results/lissa_influences.txt'


    if args.do_PBRF_search:


        print("getting all PBRF scores, EKFAC is constant")
        ekfac_refac_influences = os.getcwd() + args.best_EKFAC_path

    # pbrf_influences = os.getcwd() + '/results/PBRF_influence_scores_random_scaling_0.001_epsilon_120000.txt'

        matching_files = [filename for filename in os.listdir(os.getcwd() + '/results/') if filename.startswith('PBRF_influence_scores_random_scaling_')]

        all_corr_data = []
        column_names = ["scaling", "downweight", "corr_ekfac", "corr_lissa"]

        for filename in matching_files:

            try:
                print("the filename is {}".format(filename))
                scaling = filename.split("_")[-3]
                downweight = filename.split("_")[-1][:-4]
                pbrf_influences = os.getcwd() + '/results/{}'.format(filename)
                _ = influence_correlation(lissa_influences, ekfac_refac_influences)
                corr1 = influence_correlation(pbrf_influences, ekfac_refac_influences)
                corr2 = influence_correlation(pbrf_influences, lissa_influences)
                all_corr_data.append([scaling, downweight, corr1, corr2])
                print("\n")
            except Exception as e:
                print(e)

            df_hyper = pd.DataFrame(data = all_corr_data, columns=column_names)
            df_hyper.to_csv('random_search_pbrf_results.csv', index=False)

    elif args.do_EKFAC_search :

        pbrf_influences = os.getcwd() + args.best_PBRF_path

        all_corr_data = []

        column_names = ["scaling", "corr_pbrf", "corr_lissa"]
        matching_files = [filename for filename in os.listdir(os.getcwd() + '/results/') if filename.startswith('ekfac_refactored_influences_fc2_scaling_')]


        for filename in matching_files:

            try:
                print("the filename is {}".format(filename))
                downweight = filename.split("_")[-1][:-4]
                ekfac_refac_influences = os.getcwd() + '/results/{}'.format(filename)
                corr1 = influence_correlation(lissa_influences, ekfac_refac_influences)
                corr2 = influence_correlation(pbrf_influences, ekfac_refac_influences)
                corr3 = influence_correlation(pbrf_influences, lissa_influences)
                all_corr_data.append([downweight, corr2, corr1])
                print("\n")
            except Exception as e:
                print(e)

            df_hyper = pd.DataFrame(data = all_corr_data, columns=column_names)
            df_hyper.to_csv('random_search_ekfac_results.csv', index=False)







