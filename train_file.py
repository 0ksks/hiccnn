import os

import torch
import wandb

from warnings import filterwarnings
from typing import Literal
from data.CUB_200_2011 import (
    get_dataset as get_cub_dataset,
    single_category_dataset as single_category_cub_dataset
)
from data.voc2010_crop import get_dataset as get_voc_dataset
from data.mpii_human_pose import (
    get_dataset as get_mpii_dataset,
    single_category_dataset as single_category_mpii_dataset
)
from data.human_interaction_image import get_dataset as get_hii_dataset
from global_variable import get_config_value, parse_config_path, RUN_NAME

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

filterwarnings("ignore")


def load_datasets(dataset_name: Literal[
    "cub_200_2011", "voc_2010_crop", "mpii_human_pose.cut", "mpii_human_pose.ori", "human_interaction_image"],
                  **kwargs):
    dataset_path = parse_config_path(
        get_config_value("dataset.root") +
        get_config_value(f"dataset.{dataset_name}")
    )

    if dataset_name == "cub_200_2011":
        datasets = get_cub_dataset(dataset_path)
        for idx, dataset in enumerate(datasets):
            datasets[idx] = single_category_cub_dataset(dataset)
    elif dataset_name == "voc_2010_crop":
        datasets = get_voc_dataset(dataset_path, dataset_name="bird", **kwargs)
    elif "mpii_human_pose" in dataset_name:
        datasets = get_mpii_dataset(dataset_path, **kwargs)
        for idx, dataset in enumerate(datasets):
            datasets[idx] = single_category_mpii_dataset(dataset)
    elif dataset_name == "human_interaction_image":
        datasets = get_hii_dataset(dataset_path)
    else:
        raise Exception("dataset not supported")
    train_len = len(datasets[0])
    test_len = 0 if len(datasets) == 1 else len(datasets[1])
    print(f"dataset `{dataset_name}` loaded, train length: {train_len}, test length: {test_len}")

    return datasets


if __name__ == '__main__':
    OPTIMIZER = torch.optim.Adam
    CLASSIFICATION_LR = 1e-4
    CLUSTER_LR = 1e-6
    EPOCHS = 100
    BATCH_SIZE = 16

    from train.vgg_16_bn import load_vgg_16_bn, train_vgg_16_bn

    wandb_log_dir = parse_config_path(get_config_value("wandb.root"))
    if not os.path.exists(wandb_log_dir):
        os.makedirs(wandb_log_dir)
    model = load_vgg_16_bn(weight_path="vgg_16_bn", num_classes=2)
    datasets = load_datasets("voc_2010_crop")
    wandb.login(key=get_config_value("wandb.api_key"), force=True)
    wandb.init(
        name=RUN_NAME,
        dir=wandb_log_dir
    )
    train_vgg_16_bn(
        model=model,
        num_classes=2,
        overwrite_classifier=True,
        optimizer=OPTIMIZER,
        lr_scheduler=torch.optim.lr_scheduler.StepLR,
        optimizer_params={
            "classification_lr": CLASSIFICATION_LR,
            "cluster_lr": CLUSTER_LR,
        },
        lr_scheduler_params={
            "step_size": EPOCHS,
            "gamma": 0.6,
        },
        lightning_scheduler_params={
            "interval": "epoch",
            "frequency": 1,
        },
        datasets=datasets,
        subset_size=32,
        batch_size=BATCH_SIZE,
        num_workers=0,
        center_num=5,
        classification_loss_patience=-1,
        accuracy_threshold=0.8,
        cluster_interval=1,
        cluster_loss_factor=100,
        cluster_stop_epoch=0,
        hierarchical_loss_factor=1e-3,
        log_tmp_output_every_step=0,
        log_tmp_output_every_epoch=10,
        save_feature_map=0,
        save_pth=1,
        save_pth_path="",
        save_pth_name="",
        max_epochs=EPOCHS,
        callbacks=None,
        wandb_log_dir=wandb_log_dir,
        run_name=RUN_NAME
    )
