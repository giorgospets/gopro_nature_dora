import os
import cv2
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as T
from torchvision import transforms as pth_transforms
from torchvision import datasets

from torch.utils.data import DataLoader, random_split
torch.manual_seed(0)


class CustomIntelDataset(Dataset):
    """Custom dataset for the Nature dataset"""

    label_mapping = {
        "buildings": 0,
        "forest": 1,
        "glacier": 2,
        "mountain": 3,
        "sea": 4,
        "street": 5
    }
    
    classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.images, self.labels = self.get_images(dataset_path)

    def __getitem__(self, index: int):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

    def get_images(self, directory):
        images = []
        labels = []

        for label_str in os.listdir(directory):
            label_path = os.path.join(directory, label_str)
            if not os.path.isdir(label_path):
                continue  # Skip non-directory files

            label = self.label_mapping.get(label_str, -1)
            if label == -1:
                continue  # Ignore unknown labels

            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)  # Read image (BGR format)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                image = cv2.resize(image, (224, 224))
                
                images.append(image)
                labels.append(label)

        return images, labels


def compute_mean_std(dataset):
    """Compute mean and std using PyTorch built-in functions."""
    all_images = torch.cat([T.ToTensor()(img).unsqueeze(0) for img in dataset.images], dim=0)
    return all_images.mean(dim=(0, 2, 3)), all_images.std(dim=(0, 2, 3))


def load_intel_dataset():
    base_path = "./data/nature_dataset"
    train_path = os.path.join(base_path, "seg_train")
    test_path = os.path.join(base_path, "seg_test")
    
    mean = (0.4302, 0.4572, 0.4536)
    std = (0.2608, 0.2590, 0.2906)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    train_dataset = CustomIntelDataset(train_path, transform=transform)
    test_dataset = CustomIntelDataset(test_path, transform=transform)
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


def load_cifar10_dataset():
    mean = (0.4914, 0.4822, 0.4465)  
    std = (0.2023, 0.1994, 0.2010)
    
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
    
    test_dataset = datasets.CIFAR10(root=dataset_path, train=False, 
                                    download=False, transform=val_transform)
    
    return train_dataset, val_dataset, test_dataset


def load_datasets(dataset_name: str):
    if dataset_name == "cifar10":
        return load_cifar10_dataset()
    elif dataset_name == "intel":
        return load_intel_dataset()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
