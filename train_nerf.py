import argparse
import importlib
import os
import random
import shutil

import numpy as np
import torch
import torch.optim as optim
import yaml
from omegaconf import OmegaConf


def merge_config_file(config, config_path, allow_invalid=False):
    """
    Load yaml config file if specified and merge the arguments
    """
    if config_path is not None:
        with open(config_path, "r") as config_file:
            new_config = yaml.safe_load(config_file)
        invalid_args = list(set(new_config.keys()) - set(config.keys()))
        if invalid_args and not allow_invalid:
            raise ValueError(f"Invalid args {invalid_args} in {config_path}.")
        config.update(new_config)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def file_backup(opt, workspace):
    dir_list = ["./", "./" + opt.module]
    os.makedirs(os.path.join(workspace, "recording"), exist_ok=True)
    for dir_name in dir_list:
        cur_dir = os.path.join(workspace, "recording", dir_name)
        os.makedirs(cur_dir, exist_ok=True)
        files = os.listdir(dir_name)
        for f_name in files:
            if f_name[-3:] == ".py":
                shutil.copyfile(
                    os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name)
                )

    shutil.copyfile(opt.config, os.path.join(workspace, "config.yaml"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="Path to `config.yaml`.")
    args = parser.parse_args()

    default_config_path = "./configs/default.yaml"
    with open(default_config_path, "r") as config_file:
        opt = yaml.safe_load(config_file)
    opt["config"] = args.config if args.config else default_config_path
    merge_config_file(opt, args.config, allow_invalid=True)
    opt = OmegaConf.create(opt)
    print(opt)

    seed_everything(opt.seed)
    config_name = os.path.splitext(os.path.basename(opt.config))[0]
    workspace = os.path.join("logs", config_name, opt.module)

    # freeze config
    os.makedirs(workspace, exist_ok=True)
    file_backup(opt, workspace)

    # import backend
    Dataset = getattr(
        importlib.import_module(f"{opt.module}.atlantic_datasets"), "get_dataset"
    )
    Trainer = getattr(importlib.import_module(f"{opt.module}.utils"), "Trainer")
    Criterion = getattr(importlib.import_module(f"{opt.module}.losses"), "Criterion")
    Metric = getattr(importlib.import_module(f"{opt.module}.metrics"), "get_metrics")
    Sampler = getattr(importlib.import_module(f"{opt.module}.sampler"), "Sampler")

    # MLP backend
    if opt.backend == "torch":
        NeRFNetwork = getattr(
            importlib.import_module(f"{opt.module}.network"), "NeRFNetwork"
        )
    elif opt.backend == "tcnn":
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        NeRFNetwork = getattr(
            importlib.import_module(f"{opt.module}.network_tcnn"), "NeRFNetwork"
        )
    elif opt.backend == "cuda":
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        NeRFNetwork = getattr(
            importlib.import_module(f"{opt.module}.network_ff"), "NeRFNetwork"
        )
    else:
        raise Exception("Unsupported backend!")

    train_dataset, valid_dataset, _ = Dataset(
        bound=opt.bound,
        **opt.data,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

    model = NeRFNetwork(
        encoding="hashgrid"
        if not opt.extrinsic.optimize_extrinsics
        else "annealable_hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        n_views=train_dataset.n_views,
        **opt.network,
        **opt.extrinsic,
    )
    print(model)

    # Train
    if opt.renderer.z_far <= 0:
        opt.renderer.z_far = float(train_dataset.depth_scale)

    criterion = Criterion(opt.criterion)
    metrics = Metric(opt.metrics)
    sampler = Sampler(**opt.sampler, class_colors=train_dataset.class_colors)

    # Optimizer
    def optimizer(model):
        param_groups = [
            {"name": "encoding", "params": list(model.encoder.parameters())},
            {
                "name": "net",
                "params": list(model.sigma_net.parameters())
                + list(model.color_net.parameters()),
                "weight_decay": opt.optimizer.weight_decay,
            },
        ]
        if opt.extrinsic.optimize_extrinsics:
            param_groups.append(
                {
                    "name": "extrinsics",
                    "params": list(model.extrinsic_optimizer.parameters()),
                    "lr": opt.optimizer.learning_rate_pose,
                }
            )
        return torch.optim.Adam(
            param_groups, lr=opt.optimizer.learning_rate, betas=(0.9, 0.99), eps=1e-15
        )

    # Scheduler
    lr_lambda = lambda epoch: opt.optimizer.lr_scheduler_gamma ** (
        min(
            opt.optimizer.lr_scheduler_steps,
            epoch / opt.optimizer.lr_scheduler_step_size,
        )
    )
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = Trainer(
        name="ngp",
        conf=opt.renderer,
        model=model,
        workspace=workspace,
        optimizer=optimizer,
        criterion=criterion,
        metrics=metrics,
        ema_decay=0.95,
        fp16=opt.fp16,
        lr_scheduler=scheduler,
        sampler=sampler,
        use_checkpoint="latest",
        eval_interval=opt.eval_interval,
        depth_scale=train_dataset.depth_scale,
    )

    trainer.train(train_loader, valid_loader, opt.num_epochs)
