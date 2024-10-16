import os
from typing import Literal

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class HIIDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.image_folder = ImageFolder(root=root, transform=self.transform)

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, label = self.image_folder[idx]
        return image, torch.tensor(label).long()


def get_dataset(root: str, split: Literal["train", "test"] = None, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    train_path = os.path.join(root, 'train', "train")
    test_path = os.path.join(root, 'test', "test")
    if split == "train":
        train_dataset = ImageFolder(root=train_path, transform=transform)
        return [train_dataset, ]
    elif split == "test":
        test_dataset = ImageFolder(root=test_path, transform=transform)
        return [test_dataset, ]
    else:
        train_dataset = ImageFolder(root=train_path, transform=transform)
        test_dataset = ImageFolder(root=test_path, transform=transform)
        return [train_dataset, test_dataset]
