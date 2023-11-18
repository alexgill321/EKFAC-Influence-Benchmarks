import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the transformation to flatten the images
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

# Download MNIST dataset and create DataLoader
train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
copy_train_ds = train_dataset
# Split the training dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


# Function to add Gaussian noise to training examples
def add_gaussian_noise(data, mean=0, std=0.1):
    noisy_data = data + torch.randn_like(data) * std + mean
    return noisy_data

# Function to plot images
def plot_images(original_images, noisy_images, num_examples=5):
    for i in range(min(num_examples, original_images.size(0))):
        original_image = original_images[i].view(28, 28).numpy()
        noisy_image = noisy_images[i].view(28, 28).numpy()

        plt.subplot(2, num_examples, i + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title(f'Original {i + 1}')
        plt.axis('off')

        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.imshow(noisy_image, cmap='gray')
        plt.title(f'Noisy {i + 1}')
        plt.axis('off')

    plt.show()

# Function to loop through noisy examples
def noisy_examples(loader, num_examples=5, noise_mean=0, noise_std=0.1):
    for batch_idx, (inputs, labels) in enumerate(loader):
        noisy_inputs = add_gaussian_noise(inputs, noise_mean, noise_std)
        original_images = inputs[:num_examples]
        noisy_images = noisy_inputs[:num_examples]

        plot_images(original_images, noisy_images, num_examples)

        break  # Comment this line to loop through all batches

# Example usage:
noisy_examples(train_loader, num_examples=5)
