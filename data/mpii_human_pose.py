import cv2
import numpy as np
import os
import pandas as pd

from typing import Literal

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MPIIHumanPoseDatasetProcessor:
    def __init__(self, path):
        self.release = loadmat(path)["RELEASE"][0, 0]
        self.anno_list: np.ndarray = self.release["annolist"].flatten()
        self.img_train: np.ndarray = self.release["img_train"].flatten()
        self.version: np.ndarray = self.release["version"].flatten()
        self.single_person: np.ndarray = self.release["single_person"].flatten()
        self.act: np.ndarray = self.release["act"].flatten()
        self.video_list: np.ndarray = self.release["video_list"].flatten()

        self.dataframe = pd.DataFrame()

    def handle_anno_list(self):
        anno_list_processor = AnnoListProcessor(self.anno_list)
        anno_list_processor.handle_image()
        self.dataframe = pd.concat([self.dataframe, anno_list_processor.dataframe], axis=1)

    def handle_img_train(self):
        img_train = self.img_train.flatten()
        self.dataframe["train"] = img_train

    def handle_action(self):
        act = self.act.flatten()
        act_name = act["act_name"]
        cat_name = act["cat_name"]
        act_id = act["act_id"]

        def blank_filter(li):
            if len(li):
                return li[0]
            return ""

        act_name = list(map(blank_filter, act_name))
        cat_name = list(map(blank_filter, cat_name))
        act_id = list(map(lambda item: item.item(), act_id))

        self.dataframe["action"] = act_name
        self.dataframe["category"] = cat_name
        self.dataframe["action_id"] = act_id


class AnnoListProcessor:
    def __init__(self, anno_list: np.ndarray):
        self.anno_list = anno_list
        self.image = anno_list["image"]
        self.anno_rect = anno_list["annorect"]
        self.frame_sec = anno_list["frame_sec"]
        self.vid_idx = anno_list["vididx"]
        self.dataframe = pd.DataFrame()

    def handle_image(self):
        image_list = self.image
        image_list = image_list.flatten()
        image_list = list(map(lambda item: item.item()[0].item(), image_list))
        self.dataframe["path"] = image_list


class MPIIHumanPoseDataset(Dataset):
    def __init__(self, root, label: Literal["category", "action"], split: Literal["train", "test"],
                 single: bool, process=False):
        self.root = root
        self.label = label
        self.split = split
        integrity = self._check_integrity(single)
        if process or not integrity:
            if not integrity:
                print("datasets haven't processed, processing")
            if process:
                print("force to reprocess, processing")
            self._process_mat(label if single else None)
        self.dataframe = self._load_dataframe(single)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _check_integrity(self, single: bool):
        single_file = "_single" if single else ""
        train_path = os.path.join(self.root, f"train{single_file}.csv")
        test_path = os.path.join(self.root, f"test{single_file}.csv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            return True
        return False

    def _process_mat(self, label: Literal["category", "action"] = None):
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        processor = MPIIHumanPoseDatasetProcessor(os.path.join(self.root, "mpii_human_pose_v1_u12_1.mat"))
        processor.handle_anno_list()
        processor.handle_img_train()
        processor.handle_action()
        df = processor.dataframe
        train_set = df[df["train"] == 1].drop(columns=["train"])
        train_set['category_id'] = label_encoder.fit_transform(train_set['category'])
        test_set = df[df["train"] == 0].drop(columns=["train"])
        test_set['category_id'] = -1
        if label is not None:
            most_common_label = train_set[f"{label}_id"].value_counts().index[:2].values
            print(most_common_label)
            label0, label1 = most_common_label
            train_set = train_set[
                (train_set[f"{label}_id"] == label0) | (train_set[f"{label}_id"] == label1)
                ]
            train_set.loc[train_set[f"{label}_id"] == label0, f"{label}_id"] = 0
            train_set.loc[train_set[f"{label}_id"] == label1, f"{label}_id"] = 1
        single_file = "_single" if label else ""
        train_set.to_csv(os.path.join(self.root, f"train{single_file}.csv"))
        test_set.to_csv(os.path.join(self.root, f"test{single_file}.csv"))

    def _load_dataframe(self, single: bool):
        single_file = "_single" if single else ""
        if self.split == "train":
            return pd.read_csv(os.path.join(self.root, f"train{single_file}.csv"), index_col=False)
        if self.split == "test":
            return pd.read_csv(os.path.join(self.root, f"test{single_file}.csv"), index_col=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe["path"][idx]
        try:
            image = cv2.imread(str(os.path.join(self.root, "images", image_path)))
        except Exception:
            raise Exception
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        label = self.dataframe[f"{self.label}_id"][idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataset(path, label: Literal["category", "action"], single: bool, split: Literal["train", "test"] = None):
    if split == "train":
        dataset = MPIIHumanPoseDataset(path, label, "train", single)
        return [dataset, ]
    elif split == "test":
        dataset = MPIIHumanPoseDataset(path, label, "test", single)
        return [dataset, ]
    else:
        train_dataset = MPIIHumanPoseDataset(path, label, "train", single)
        test_dataset = MPIIHumanPoseDataset(path, label, "test", single)
        return [train_dataset, test_dataset]


def single_category_dataset(dataset: MPIIHumanPoseDataset,
                            label: Literal["category", "action"]) -> MPIIHumanPoseDataset:
    df = dataset.dataframe
    if label == "category":
        df = df[(df['category_id'] == 11) | (df['category_id'] == 15)]
        df.loc[df['category_id'] == 11, "category_id"] = 0
        df.loc[df['category_id'] == 15, "category_id"] = 1
    if label == "action":
        df = df[(df['action_id'] == 53) | (df['action_id'] == 378)]
        df.loc[df['action_id'] == 53, "action_id"] = 0
        df.loc[df['action_id'] == 378, "action_id"] = 1
    dataset.dataframe = df
    return dataset
