import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def load_data(store_mnist='../data'):
    # Define the transformation to flatten the images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

    # Download MNIST dataset and create DataLoader
    train_dataset = datasets.MNIST(root=store_mnist, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=store_mnist, train=False, transform=transform, download=True)

    # Split the training dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader

# Define a neural network model with a transformer encoder layer and a linear layer
class TransformerModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_heads=1, num_layers=1, output_size=10):
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads),
            num_layers=num_layers
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        x = self.fc(x)  # Use only the last transformer output
        # x = self.relu(x)
        # x = self.output_layer(x)
        return x

# Instantiate the model, loss function, and optimizer
model = TransformerModel()
criterion = nn.CrossEntropyLoss()
#mse_criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
def train(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Compute cross-entropy loss
            loss_ce = criterion(outputs, labels)
            
            # Compute mean squared error
            #loss_mse = mse_criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
            
            # Combine the two losses
            loss = loss_ce #+ loss_mse
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        # Print validation loss during training
        val_loss, val_acc = validate(model, val_loader, criterion)#, mse_criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%")
        torch.save(model.state_dict(), f'../models/checkpoints/checkpoint_{epoch+1}_transformer_trained_model.pth')
    # Save the trained model
    torch.save(model.state_dict(), '../models/transformer_trained_model.pth')
    print("Trained model saved.")

# Validation loop
def validate(model, val_loader, criterion):#, mse_criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            
            # Compute cross-entropy loss
            loss_ce = criterion(outputs, labels)
            
            # Compute mean squared error
            #loss_mse = mse_criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
            
            # Combine the two losses
            loss = loss_ce #+ loss_mse
            
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# Testing loop
def test(model, test_loader, criterion):#, mse_criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            # Compute cross-entropy loss
            loss_ce = criterion(outputs, labels)
            
            # Compute mean squared error
            #loss_mse = mse_criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
            
            # Combine the two losses
            loss = loss_ce #+ loss_mse
            
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")
    return avg_loss, accuracy

def get_model():
    model = TransformerModel()
    criterion = nn.CrossEntropyLoss()
    #mse_criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return model, criterion, optimizer


# Function to load the model from a saved state
def load_model(model, filepath='../models/transformer_trained_model.pth'):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_data()
    summary(model, (64, 784))
    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, epochs=10)

    # Test the model
    test_loss, test_accuracy = test(model, test_loader, criterion)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy * 100:.2f}%")
