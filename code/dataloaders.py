
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset

def get_mnist_dataloaders(batch_size=64):
    
    Transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.MNIST('../mnist_train', train=True, download=True,
                                transform=Transform)
    test_data = datasets.MNIST('../mnist_test', train=False, download= True,
                               transform=Transform)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader



def get_cifar_dataloaders(batch_size=64):

    Transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.CIFAR10('../cifar_train', train=True, download=True,
                                transform=Transforms)
    test_data = datasets.CIFAR10('../cifar_test', train=False, download= True,
                               transform=Transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

    


