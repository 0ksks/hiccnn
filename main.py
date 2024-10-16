import torch
import os
import wandb
from warnings import filterwarnings

from global_variable import get_config_value, parse_config_path, RUN_NAME

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

filterwarnings("ignore")


def objective(config, datasets, wandb_log_dir, run_name):
    optimizer = config.optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD
    train_vgg_16_bn(
        #  ========================= network =========================
        model=model,
        num_classes=2,
        overwrite_classifier=True,
        #  ========================= optimization =========================
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.StepLR,
        optimizer_params={
            "classification_lr": config.classification_lr,
            "cluster_lr": config.cluster_lr,
        },
        lr_scheduler_params={
            "step_size": config.epochs,
            "gamma": 0.6,
        },
        lightning_scheduler_params={
            "interval": "epoch",
            "frequency": 1,
        },
        #  ========================= dataset =========================
        datasets=datasets,
        subset_size=8,
        batch_size=4,
        num_workers=0,
        #  ========================= loss =========================
        center_num=5,
        classification_loss_patience=10,
        accuracy_threshold=0.9,
        cluster_interval=1,
        cluster_loss_factor=1e-1,
        cluster_stop_epoch=0,
        hierarchical_loss_factor=1e-3,
        max_epochs=config.epochs,
        callbacks=None,
        #  ========================= log =========================
        log_tmp_output_every_step=0,
        log_tmp_output_every_epoch=0,
        save_feature_map=0,
        save_pth=0,
        save_pth_path="",
        save_pth_name="",
        wandb_log_dir=wandb_log_dir,
        run_name=run_name
    )


# def cli(
#         subset_size: int = typer.Option(
#             32,
#             "--subset-size", "-ss",
#             help="the subset size of the training set, set `0` to use all",
#             prompt="subset size(`0` to use all)"
#         ),
#         batch_size: int = typer.Option(
#             4,
#             "--batch-size", "-bs",
#             help="the batch size for training",
#             prompt="batch size"
#         ),
#         num_workers: int = typer.Option(
#             0,
#             "--num-workers", "-nw",
#             help="the number of workers for training, set `0` to use half",
#             prompt="num workers(`0` to use half)"
#         ),
#         center_num: int = typer.Option(
#             5,
#             "--center-num", "-cn",
#             help="the number of cluster center when clustering",
#             prompt="center num"
#         ),
#         cluster_interval: int = typer.Option(
#             1,
#             "--cluster-loss-interval", "-cli",
#             help="the interval between clustering loss updates",
#             prompt="cluster loss interval"
#         ),
#         cluster_loss_factor: float = typer.Option(
#             1e-1,
#             "--cluster-loss-factor", "-clf",
#             help="the loss factor on clustering loss when adding up all losses",
#             prompt="cluster loss factor"
#         ),
#         cluster_stop_epoch: int = typer.Option(
#             200,
#             "--cluster-stop-epoch", "-cse",
#             help="stop clustering after this epoch",
#             prompt="cluster stop epoch"
#         ),
#         hierarchical_loss_factor: float = typer.Option(
#             0.1,
#             "--hierarchical-loss-factor", "-hlf",
#             help="the loss factor on hierarchical clustering loss when adding up all losses",
#             prompt="hierarchical loss factor"
#         ),
#         log_tmp_output_every_step: int = typer.Option(
#             0,
#             "--log-tmp-output-every-step", "-step",
#             help="to log tmp output every step, set `0` to disable log",
#             prompt="log tmp output every step(`0` to disable)"
#         ),
#         log_tmp_output_every_epoch: int = typer.Option(
#             0,
#             "--log-tmp-output-every-epoch", "-epoch",
#             help="to log tmp output every epoch, set `0` to disable log",
#             prompt="log tmp output every epoch(`0` to disable)"
#         ),
#         save_pth: int = typer.Option(
#             0,
#             "--save-pth", "-sw",
#             help="to save model weights, set `0` to disable",
#             prompt="save model weights or not(`0` to disable)"
#         ),
#         save_pth_path: str = typer.Option(
#             "0",
#             "--save-pth-path", "-swp",
#             help="where to save model weights, set `0` to use default dir",
#             prompt=f"save path(`0` to default `{parse_config_path(get_config_value("weight.root"))}`)"
#         ),
#         save_pth_name: str = typer.Option(
#             "0",
#             "--save-pth-name", "-swn",
#
#             help="name of the model weights, set `0` to use default name",
#             prompt="save name(`0` to default `model_${global_step}.pth`)"
#         ),
#         save_feature_map: int = typer.Option(
#             0,
#             "--save-feature-map", "-epoch",
#             help="to log feature map every epoch, set `0` to disable log",
#             prompt="log feature map every epoch(`0` to disable)"
#         ),
#         max_epochs: int = typer.Option(
#             100,
#             "--max-epoch", "-max",
#             help="the maximum epochs for training",
#             prompt="max epochs"
#         ),
#         wandb_online: int = typer.Option(
#             0,
#             "--wandb-online", "-sync",
#             help="whether upload to wandb",
#             prompt="wandb online"
#         ),
#         run_name: str = typer.Option(
#             RUN_NAME,
#             "--run-name", "-rn",
#             help="the name of the run, default is datetime now",
#             prompt="run name"
#         )
# ):
#     main(
#         subset_size, batch_size, num_workers,
#         center_num, cluster_interval, cluster_loss_factor, cluster_stop_epoch,
#         hierarchical_loss_factor,
#         log_tmp_output_every_step, log_tmp_output_every_epoch,
#         save_pth, save_pth_path, save_pth_name, save_feature_map,
#         max_epochs, wandb_online, run_name
#     )


if __name__ == "__main__":
    from train.vgg_16_bn import load_vgg_16_bn, train_vgg_16_bn
    from train_file import load_datasets
    import yaml

    with open("sweep.yaml", encoding="utf-8") as file:
        sweep_config = yaml.safe_load(file)
    WANDB_ONLINE = 0
    if WANDB_ONLINE == 0:
        mode = "offline"
    else:
        mode = "online"
    wandb_log_dir = parse_config_path(get_config_value("wandb.root"))
    if not os.path.exists(wandb_log_dir):
        os.makedirs(wandb_log_dir)
    model = load_vgg_16_bn(weight_path="vgg_16_bn", num_classes=2)
    datasets = load_datasets("voc_2010_crop")


    def main():
        wandb.login(key=get_config_value("wandb.api_key"))
        wandb.init(
            name=RUN_NAME,
            dir=wandb_log_dir,
            mode=mode,
            config=sweep_config
        )
        objective(wandb.config, datasets, wandb_log_dir, RUN_NAME)


    sweep_id = wandb.sweep(sweep_config, project=get_config_value("wandb.project"))
    wandb.agent(sweep_id, function=main, project=get_config_value("wandb.project"))
