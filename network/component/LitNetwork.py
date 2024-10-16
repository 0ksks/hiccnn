import os
from abc import abstractmethod
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


class LitModel(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            optimizer: type,
            lr_scheduler: type,
            optimizer_params: dict,
            lr_scheduler_params: dict,
            lightning_scheduler_params: dict,
            save_pth: int,
            save_pth_path: str,
            save_pth_name: str,
            classification_loss_patience: int,
            accuracy_threshold: float,
            cluster_interval: int = 1,
            log_tmp_output_every_step: int = None,
            log_tmp_output_every_epoch: int = None,
            example_input: torch.Tensor = None
    ):
        """
        optimizer_params: `classification_lr` `cluster_lr`
        """
        super().__init__()
        #  save params
        self.model = model.to(self.device)
        self.cluster_interval = cluster_interval
        self.log_tmp_output_every_step = log_tmp_output_every_step
        self.log_tmp_output_every_epoch = log_tmp_output_every_epoch
        self.example_input = example_input
        self.save_pth = save_pth
        self.save_pth_path = save_pth_path
        self.save_pth_name = save_pth_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.optimizer_params = optimizer_params
        self.lr_scheduler_params = lr_scheduler_params
        self.lightning_scheduler_params = lightning_scheduler_params

        #  save accuracy
        self.current_accuracy = torch.tensor(0.0)

        #  config classification
        self.classification_loss_patience = classification_loss_patience
        self.classification_loss_counter = 0
        self.accuracy_threshold = accuracy_threshold
        self.start_cluster = False

        #  store outputs intercepted by hooks
        self.intercept_output: dict[str, torch.Tensor] = {}
        self.grid_images: dict[str, np.ndarray] = {}

        #  lock, to avoid hooking recursively
        self.log_lock = False

        #  register hooks for Conv2d layers
        for name, layer in model.named_modules():
            flag, formatted_name = self.conv_2d_filter(name, layer)
            if flag:
                layer.register_forward_hook(self.hook_feature_map(formatted_name, layer))

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        self.log_lock = True
        outputs = self.model(inputs)
        self.log_lock = False
        log_dict = self.training_step_loss_fn(inputs, outputs, labels)
        log_dict.update({"learning_rate": self.optimizer.param_groups[0]["lr"]})

        train_loss = log_dict["train_loss"]

        self.logger.experiment.log(log_dict)

        #  intermittently log feature maps
        if self.log_tmp_output_every_step and self.global_step % self.log_tmp_output_every_step == 0:
            self.log_tmp_output()

        return train_loss

    def on_train_epoch_end(self) -> None:
        #  calculate accuracy every single epoch
        correct_sum = 0
        samples_sum = 0
        with torch.no_grad():
            for inputs, labels in self.train_dataloader():
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                pred = torch.argmax(outputs, dim=1)
                correct = (pred == labels).sum()
                samples = labels.size(0)
                correct_sum += correct.item()
                samples_sum += samples
        accuracy = correct_sum / samples_sum
        self.current_accuracy = accuracy
        self.logger.experiment.log({"accuracy": accuracy})
        #  update counter
        if self.current_accuracy < self.accuracy_threshold:
            self.classification_loss_counter = 0
        else:
            self.classification_loss_counter += 1
        self.start_cluster = self.classification_loss_counter > self.classification_loss_patience
        #  intermittently log feature maps
        if self.log_tmp_output_every_epoch and self.current_epoch % self.log_tmp_output_every_epoch == 0:
            self.log_tmp_output()

    def configure_optimizers(self):
        optimizer_params = self.optimizer_params
        classification_lr = optimizer_params.pop("classification_lr")
        cluster_lr = optimizer_params.pop("cluster_lr")
        if self.start_cluster:
            optimizer_params["lr"] = cluster_lr
        else:
            optimizer_params["lr"] = classification_lr
        self.optimizer = self.optimizer(self.model.parameters(), **optimizer_params)
        self.lr_scheduler = self.lr_scheduler(self.optimizer, **self.lr_scheduler_params)
        self.lightning_scheduler_params = self.lightning_scheduler_params
        self.lightning_scheduler_params.update({'scheduler': self.lr_scheduler})

        if self.start_cluster:
            return [self.optimizer], [self.lightning_scheduler_params]
        else:
            return [self.optimizer]

    @abstractmethod
    def conv_2d_filter(self, name: str, layer: nn.Module) -> tuple[bool, str]:
        """feature map hook interface"""
        pass

    @abstractmethod
    def hook_feature_map(self, name: str, layer: nn.Module) -> tuple[bool, str]:
        pass

    @abstractmethod
    def training_step_loss_fn(
            self, inputs: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        loss function interface, key `train_loss` required
        :param inputs: (B, C, H, W)
        :param outputs: (B, classes)
        :param labels: (B)
        """
        pass

    @abstractmethod
    def log_tmp_output(self):
        pass

    def add_intercept_output(self, from_key: str, to_key: str):
        """
        get outputs by the `from_key` from NN then add it to the `to_key` in `intercept_output`
        """
        layer = dict([*self.model.named_modules()])[from_key]

        def hook(module, input, output):
            self.intercept_output[to_key] = output

        layer.register_forward_hook(hook)

    def forward(self, x):
        return self.model(x)

    def on_train_end(self) -> None:
        if self.save_pth:
            name = self.save_pth_name if self.save_pth_name else f"model_{self.current_epoch}.pth"
            torch.save(self.model.state_dict(), os.path.join(self.save_pth_path, name))
