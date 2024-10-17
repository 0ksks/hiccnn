import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.datasets.folder import default_loader
from typing import Literal


class MPIIHumanPoseDataset(Dataset):
    def __init__(self, root, label_key: Literal["code_0", "code_1"] = None, loader=default_loader, transform=None):
        self.root = Path(root)
        self.label_key = label_key
        self.loader = loader
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 将图像缩放到 224x224
                transforms.ToTensor(),  # 转换为 Tensor
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和标准差进行标准化
            ])
        self.df = self.__load_dataset()

    def __load_dataset(self) -> pd.DataFrame:
        pkl_path = self.root / "fiftyone_dataset.pkl"
        df = pd.read_pickle(pkl_path)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not self.label_key:
            raise Exception("must provide label_key")
        path = self.df.iloc[idx]["path"]
        path = Path(path)
        path = self.root / "image" / path
        image = self.loader(str(path))
        image = self.transform(image)
        target = torch.tensor(self.df.iloc[idx][self.label_key]).long()
        return image, target


def get_dataset(root, label_key: Literal["code_0", "code_1"] = None):
    return [MPIIHumanPoseDataset(root, label_key=label_key), ]


def single_category_dataset(dataset: MPIIHumanPoseDataset):
    df = dataset.df
    filter_keys = df[dataset.label_key].value_counts().index[:2].values
    df = df[(df[dataset.label_key] == filter_keys[0]) | (df[dataset.label_key] == filter_keys[1])]
    df.loc[df[dataset.label_key] == filter_keys[0], dataset.label_key] = 0
    df.loc[df[dataset.label_key] == filter_keys[1], dataset.label_key] = 1
    dataset.df = df
    return dataset
