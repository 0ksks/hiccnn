import random
from typing import Union

import torch
import os
import wandb
from torchvision.models import vgg16_bn
from collections import OrderedDict

from torch import nn
from torch.utils.data import Subset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn.cluster import SpectralClustering

from data import get_dataloader
from network.VGG16BN import VGG16BN
from global_variable import get_config_value, parse_config_path


def load_vgg_16_bn(weight_path, num_classes):
    vgg16 = vgg16_bn(weights=None)
    state_dict = torch.load(
        parse_config_path(
            get_config_value("weight.root") +
            get_config_value(f"weight.{weight_path}")
        ),
        weights_only=True,
        map_location=torch.device("cpu"),  # TODO gpu
    )

    if weight_path == "model_2499":  # TODO why change the original fc layers?
        vgg16.classifier[3] = torch.nn.Linear(
            in_features=4096,
            out_features=512
        )
        vgg16.classifier[6] = torch.nn.Linear(  # change to target class number
            in_features=vgg16.classifier[3].out_features,
            out_features=num_classes
        )
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.module.', '')
            new_state_dict[name] = v
        state_dict = new_state_dict

    vgg16.load_state_dict(state_dict)
    return vgg16


def train_vgg_16_bn(
        model: nn.Module,
        num_classes: int,
        overwrite_classifier: bool,
        optimizer: type,
        lr_scheduler: type,
        optimizer_params: dict,
        lr_scheduler_params: dict,
        lightning_scheduler_params: dict,
        datasets: list[torch.utils.data.Dataset],
        subset_size: int,
        batch_size: int,
        num_workers: int,
        center_num: int,
        classification_loss_patience: int,
        accuracy_threshold: float,
        cluster_interval: int,
        cluster_loss_factor: float,
        cluster_stop_epoch: int,
        hierarchical_loss_factor: float,
        log_tmp_output_every_step: int,
        log_tmp_output_every_epoch: int,
        save_pth: int,
        save_pth_path: str,
        save_pth_name: str,
        save_feature_map: int,
        max_epochs: int,
        callbacks: Union[list, None],
        wandb_log_dir: str,
        run_name: str,
):
    wandb_logger = WandbLogger(
        project=get_config_value("wandb.project"),
        name=run_name,
        save_dir=wandb_log_dir
    )

    train_dataset = [datasets[0], ]

    if subset_size:
        train_dataset[0] = Subset(
            train_dataset[0],
            random.sample(
                range(len(train_dataset[0])),
                subset_size
            )
        )

    if not num_workers:
        num_workers = os.cpu_count() // 2

    if not log_tmp_output_every_step:
        log_tmp_output_every_step = None
    if not log_tmp_output_every_epoch:
        log_tmp_output_every_epoch = None

    example_input, _ = datasets[1][0]
    example_input, _ = datasets[0][0]
    if cluster_stop_epoch == 0:
        cluster_stop_epoch = max_epochs

    if save_pth == "0":
        save_pth = None
    if save_pth_path == "0":
        save_pth_path = None

    if save_pth and not save_pth_path:
        save_pth_path = parse_config_path(get_config_value("weight.root"))

    vgg_16_bn = VGG16BN(
        model=model,
        optimizer=optimizer, lr_scheduler=lr_scheduler,
        optimizer_params=optimizer_params, lr_scheduler_params=lr_scheduler_params,
        lightning_scheduler_params=lightning_scheduler_params,
        center_num=center_num, num_classes=num_classes,
        example_input=example_input.unsqueeze(0), train_dataloader=get_dataloader(
            train_dataset,
            train_kwargs=dict(
                num_workers=num_workers,
                persistent_workers=True
            ),
            split="train",
            batch_size=batch_size
        ),
        classification_loss_patience=classification_loss_patience,
        accuracy_threshold=accuracy_threshold,
        cluster_class=SpectralClustering, cluster_interval=cluster_interval,
        cluster_loss_factor=cluster_loss_factor, cluster_stop_epoch=cluster_stop_epoch,
        hierarchical_loss_factor=hierarchical_loss_factor,
        log_tmp_output_every_step=log_tmp_output_every_step,
        log_tmp_output_every_epoch=log_tmp_output_every_epoch,
        save_pth=save_pth,
        save_pth_path=save_pth_path,
        save_pth_name=save_pth_name,
        overwrite_classifier=overwrite_classifier,
        save_feature_map=save_feature_map
    )

    wandb.log(
        {
            "cluster_loss_factor": cluster_loss_factor,
            "hierarchical_loss_factor": hierarchical_loss_factor,
        },
        step=0
    )

    trainer = Trainer(logger=wandb_logger, max_epochs=max_epochs, accelerator="auto", callbacks=callbacks)
    trainer.fit(vgg_16_bn)

    wandb.teardown()
