import argparse
import importlib
import os
import random
import shutil

import numpy as np
import torch
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--out", "-o", type=str, default="")
    args = parser.parse_args()

    default_config_path = "./configs/default.yaml"
    with open(default_config_path, "r") as config_file:
        opt = yaml.safe_load(config_file)
    opt["config"] = args.config if args.config else default_config_path
    opt["out"] = args.out
    merge_config_file(opt, args.config, allow_invalid=True)
    opt = OmegaConf.create(opt)
    print(opt)

    seed_everything(opt.seed)
    config_name = os.path.splitext(os.path.basename(opt.config))[0]
    workspace = os.path.join("logs", config_name, opt.module)

    # import backend
    Dataset = getattr(
        importlib.import_module(f"{opt.module}.atlantic_datasets"), "get_dataset"
    )
    Trainer = getattr(importlib.import_module(f"{opt.module}.utils"), "Trainer")
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

    model = NeRFNetwork(
	encoding="hashgrid"
	if not opt.extrinsic.optimize_extrinsics
	else "annealable_hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        **opt.network,
    )
    print(model)

    # Test
    metrics = Metric(opt.metrics)

    _, _, test_dataset = Dataset(
        bound=opt.bound,
        **opt.data,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    if opt.renderer.z_far <= 0:
        opt.renderer.z_far = float(test_dataset.depth_scale)

    sampler = Sampler(**opt.sampler, class_colors=test_dataset.class_colors)

    trainer = Trainer(
        name="ngp",
        conf=opt.renderer,
        model=model,
        metrics=metrics,
        workspace=workspace,
        fp16=opt.fp16,
        sampler=sampler,
        use_checkpoint="latest",
        depth_scale=test_dataset.depth_scale,
    )

    trainer.test(test_loader, alpha_premultiplied=opt.test.alpha_premultiplied)

    # Video
    video_path = os.path.join(workspace, "video.webm")
    ffmpeg_bin = "ffmpeg"
    frame_regexp = os.path.join(workspace, "results", "%04d.png")
    pix_fmt = "yuva420p"
    ffmcmd = (
        '%s -r %d -i %s -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" -c:v libvpx-vp9 -crf %d -b:v 0 -pix_fmt %s -y -an %s'
        % (ffmpeg_bin, opt.test.fps, frame_regexp, opt.test.crf, pix_fmt, video_path)
    )
    ret = os.system(ffmcmd)
    if ret != 0:
        raise RuntimeError("ffmpeg failed!")

    # Output
    if opt.out != "":
        if not opt.out.startswith("s3://"):
            if os.path.isdir(opt.out):
                opt.out = os.path.join(opt.out, f"{config_name}.webm")
            shutil.copyfile(video_path, opt.out)
        else:
            if not opt.out.endswith(".webm"):
                opt.out = os.path.join(opt.out, f"{config_name}.webm")
            osscmd = f"aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp {video_path} {opt.out}"
            ret = os.system(osscmd)
            if ret != 0:
                raise RuntimeError("oss cp failed!")
            print(f"Video path: http://oss.iap.hh-b.brainpp.cn/{opt.out[5:]}")
