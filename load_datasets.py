import torch
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, random_split
torch.manual_seed(0)

from torchvision import transforms as pth_transforms


def load_datasets():
    # Define the mean and standard deviation for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)  # Precomputed mean for CIFAR-10
    std = (0.2023, 0.1994, 0.2010)   # Precomputed std for CIFAR-10
    
    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])

    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])


    dataset_path = "./data/cifar-10"
    cifar10_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=False)

    # Split the dataset into train, val, and test sets
    num_train = 45000  
    num_val = 5000    
    num_test = 10000 

    train_dataset, val_dataset = random_split(cifar10_dataset, [num_train, num_val])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=val_transform)

    # Print dataset sizes for verification
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset
