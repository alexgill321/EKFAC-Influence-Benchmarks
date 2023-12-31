#%%
from torchvision import datasets
from torch.utils.data import Subset
import os
import ast

train_dataset = datasets.MNIST(root='../data', train=True, download=True)
test_dataset = Subset(train_dataset, range(500))

#%%
array_list = []

# Replace with the path to your top_influences.txt file
with open(os.getcwd() + '/top_influences.txt', 'r') as file:
    for _ in range(2):  # Skip the first two lines
        file.readline()

    for line in file:
        line = line.strip()
        splits = line.split(': ')
        if len(splits) != 2:
            continue
        try:
            list = ast.literal_eval(splits[1])
            array_list.append(list)
        except ValueError as e:
            continue

#%%
import matplotlib.pyplot as plt

for j, list in enumerate(array_list):
    fig, axes = plt.subplots(1, 6, figsize=(15, 3)) 
    # Iterate over image paths and axes to plot each image
    for i, (index, ax) in enumerate(zip(list, axes[:5])):
        image = train_dataset[index][0]  # Load the image
        # Display the image on the current axis
        ax.imshow(image)
        ax.axis('off')  # Turn off axis labels
        ax.set_title(f"Image {index}")
    
    image = train_dataset[j][0]
    axes[5].imshow(image)
    axes[5].axis('off')
    axes[5].set_title(f"Actual Image {j}")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    if j == 20:
        break
