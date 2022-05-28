import glob
import os
import pickle
import random
import time
import warnings
from datetime import datetime

import cv2
import mcubes
import numpy as np
import tensorboardX
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import trimesh
from rich.console import Console
from torch_ema import ExponentialMovingAverage


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def distance2depth(distance, intrinsics):
    h, w = distance.shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    u = np.expand_dims(np.arange(w), 0).repeat(h, axis=0)
    v = np.expand_dims(np.arange(h), 1).repeat(w, axis=1)
    u_u0_by_fx = (u - cx) / fx
    v_v0_by_fy = (v - cy) / fy
    depth = distance / np.sqrt(u_u0_by_fx**2 + v_v0_by_fy**2 + 1)
    return depth


# non-premultiplied -> pre-multiplied
def mix_background_color(image, bg_color=1):
    if image.shape[-1] == 4:
        image_premul = image[..., :3] * image[..., 3:] + bg_color * (1 - image[..., 3:])
        image_premul = torch.cat(
            [image_premul, image[..., 3:]], dim=-1
        )  # avoid inplace op
        return image_premul
    else:
        return image


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(
                        xs, ys, zs, indexing="ij"
                    )  # for torch < 1.10, should remove indexing='ij'
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    ).unsqueeze(
                        0
                    )  # [1, N, 3]
                    val = (
                        query_func(pts)
                        .reshape(len(xs), len(ys), len(zs))
                        .detach()
                        .cpu()
                        .numpy()
                    )  # [1, N, 1] --> [x, y, z]
                    u[
                        xi * N : xi * N + len(xs),
                        yi * N : yi * N + len(ys),
                        zi * N : zi * N + len(zs),
                    ] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = (
        vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :]
        + b_min_np[None, :]
    )
    return vertices, triangles


class Trainer(object):
    def __init__(
        self,
        name,  # name of this experiment
        conf,  # extra conf
        model,  # network
        criterion=None,  # loss function, if None, assume inline implementation in train_step
        optimizer=None,  # optimizer
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
        sampler=None,  # ray sampler
        metrics=[],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        local_rank=0,  # which GPU am I
        world_size=1,  # total num of GPUs
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,  # whether to mute all print
        fp16=False,  # amp optimize level
        eval_interval=1,  # eval once every $ epoch
        max_keep_ckpt=2,  # max num of saved ckpts in disk
        workspace="workspace",  # workspace to save logs & ckpts
        best_mode="min",  # the smaller/larger result, the better
        use_loss_as_metric=True,  # use loss as the first metirc
        use_checkpoint="latest",  # which ckpt to use at init time
        use_tensorboardX=True,  # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
        depth_scale=8.0,  # scale depth to [0, 1] when visualizing
    ):

        self.name = name
        self.conf = conf
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.depth_scale = depth_scale
        self.device = (
            device
            if device is not None
            else torch.device(
                f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            )
        )
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )
        self.model = model

        self.sampler = sampler

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        for metric in self.metrics:
            if isinstance(metric, torch.nn.Module):
                metric.to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=0.001, weight_decay=5e-4
            )  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=ema_decay
            )
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = "min"

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        self.log(
            f"[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}"
        )

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data):
        # refine poses
        data["pose"] = self.model.refine_extrinsic(data["pose"], data["camera_index"])

        # baseline correction of stereo poses
        data["pose"] = torch.bmm(data["pose"], data["pose_offset"])

        # sample rays
        rays_o, rays_d, inds = self.sampler.sample(**data)

        # gather gt
        B, H, W, C = data["image"].shape
        data["image"] = torch.gather(
            data["image"].reshape(B, -1, C), 1, torch.stack(C * [inds], -1)
        )  # [B, N, 3/4]
        if "depth" in data:
            data["depth"] = torch.gather(
                data["depth"].reshape(B, -1, 1), 1, torch.stack(1 * [inds], -1)
            )[
                :, :, 0
            ]  # [B, N]

        # train
        progress = self.global_step / self.total_steps
        preds = self.model.render(
            rays_o,
            rays_d,
            staged=False,
            alpha_premultiplied=False,
            perturb=True,
            progress=progress,
            **self.conf,
        )

        # mix random background color
        bg_color = torch.rand(3, device=self.device)  # [3], frame-wise random.
        data["image"] = mix_background_color(data["image"], bg_color)
        preds["image"] = mix_background_color(preds["image"], bg_color)

        # calculate loss
        losses = self.criterion(preds, data)

        return preds, data, losses

    def eval_step(self, data):
        # sample rays
        B, H, W, C = data["image"].shape
        rays_o, rays_d, _ = self.sampler.sample(full_grid=True, **data)

        preds = self.model.render(
            rays_o,
            rays_d,
            staged=True,
            alpha_premultiplied=False,
            perturb=False,
            **self.conf,
        )

        preds["image"] = preds["image"].reshape(B, H, W, -1)
        preds["depth"] = preds["depth"].reshape(B, H, W)
        preds["depth_variance"] = preds["depth_variance"].reshape(B, H, W)

        return preds

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, perturb=False):
        H, W = int(data["H"][0]), int(data["W"][0])  # get the target size...
        B = data["pose"].shape[0]

        rays_o, rays_d, _ = self.sampler.sample(full_grid=True, **data)

        preds = self.model.render(
            rays_o,
            rays_d,
            staged=True,
            alpha_premultiplied=False,
            perturb=perturb,
            **self.conf,
        )

        preds["image"] = preds["image"].reshape(B, H, W, -1)
        preds["depth"] = preds["depth"].reshape(B, H, W)
        preds["depth_variance"] = preds["depth_variance"].reshape(B, H, W)

        return preds

    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(
                self.workspace, "meshes", f"{self.name}_{self.epoch}.ply"
            )

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sdfs = self.model.density(pts.to(self.device))
            return sdfs

        bounds_min = torch.FloatTensor([-self.model.bound] * 3)
        bounds_max = torch.FloatTensor([self.model.bound] * 3)

        vertices, triangles = extract_geometry(
            bounds_min,
            bounds_max,
            resolution=resolution,
            threshold=threshold,
            query_func=query_func,
        )

        mesh = trimesh.Trimesh(
            vertices, triangles, process=False
        )  # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name)
            )

        self.total_epochs = max_epochs
        self.total_steps = self.total_epochs * len(train_loader)

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, bg_color=1, alpha_premultiplied=False, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, "results")

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        for metric in self.metrics:
            metric.clear()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):

                data = self.prepare_data(data)

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds = self.test_step(data)

                if alpha_premultiplied:
                    preds["image"] = mix_background_color(preds["image"], bg_color)
                    if "image" in data:
                        data["image"] = mix_background_color(data["image"], bg_color)

                frame = self.visualize(preds, data)
                path = os.path.join(save_path, f"{i:04d}.png")
                cv2.imwrite(path, frame)

                if "image" in data:
                    # Calculate metrics
                    if not alpha_premultiplied:
                        preds["image"] = mix_background_color(preds["image"], bg_color)
                        data["image"] = mix_background_color(data["image"], bg_color)

                    # mask away instance when computing loss
                    if (
                        "instance_seg" in data
                        and self.sampler.ray_sampler.mask_away_instance
                    ):
                        mask = data["instance_seg"].to(torch.float32).sum(axis=-1) != 0
                        preds["image"][mask] = 0
                        data["image"][mask] = 0

                    for metric in self.metrics:
                        metric.update(preds, data)

                    pbar_desc = ""
                    for metric in self.metrics:
                        pbar_desc += str(metric)

                pbar.set_description("" + pbar_desc)
                pbar.update(loader.batch_size)

        pbar.close()
        for metric in self.metrics:
            self.log(metric.report(), style="blue")
        self.log(f"==> Finished Test.")

    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else:  # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        )

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # update grid
        if self.model.cuda_ray:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        self.local_step = 0

        for data in loader:

            self.local_step += 1
            self.global_step += 1

            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, losses = self.train_step(data)
                loss = 0
                for l in losses.values():
                    loss += l

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                for metric in self.metrics:
                    metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    for name, l in losses.items():
                        self.writer.add_scalar(
                            f"train/{name}", l.item(), self.global_step
                        )
                    for param in self.optimizer.param_groups:
                        self.writer.add_scalar(
                            f"train/lr_{param['name']}", param["lr"], self.global_step
                        )

                pbar_desc = f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}) "
                for metric in self.metrics:
                    pbar_desc += str(metric)
                if self.scheduler_update_every_step:
                    pbar.set_description(
                        pbar_desc + f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    pbar.set_description(pbar_desc)
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            for metric in self.metrics:
                self.log(metric.report(), style="red")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="train")
                metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        results = {
            "images": [],
            "depths": [],
            "weights": [],
            "images_gt": [],
            "depths_gt": [],
            "poses": [],
        }

        with torch.no_grad():
            self.local_step = 0
            for data in loader:

                self.local_step += 1
                data = self.prepare_data(data)

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    for key, pred in preds.items():
                        pred_list = [
                            torch.zeros_like(pred).to(self.device)
                            for _ in range(self.world_size)
                        ]  # [[B, ...], [B, ...], ...]
                        dist.all_gather(pred_list, pred)
                        preds[key] = torch.cat(pred_list, dim=0)

                preds["image"] = mix_background_color(preds["image"])
                data["image"] = mix_background_color(data["image"])

                if self.local_rank == 0:
                    results = self.get_results(preds, data, results)

                    frame = self.visualize(preds, data)
                    save_path = os.path.join(
                        self.workspace,
                        "validation",
                        f"{self.name}_{self.epoch:04d}_{self.local_step:04d}.png",
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, frame)

                # mask away instance when computing loss
                if (
                    "instance_seg" in data
                    and self.sampler.ray_sampler.mask_away_instance
                ):
                    mask = data["instance_seg"].to(torch.float32).sum(axis=3) != 0
                    preds["image"][mask] = 0
                    data["image"][mask] = 0
                    preds["depth_variance"][mask] = 0
                    if "depth" in data:
                        preds["depth"][mask] = 0
                        data["depth"][mask] = 0

                losses = self.criterion(preds, data)
                loss = 0
                for l in losses.values():
                    loss += l

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, data)

                    pbar_desc = (
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}) "
                    )
                    for metric in self.metrics:
                        pbar_desc += str(metric)
                    pbar.set_description(pbar_desc)
                    pbar.update(loader.batch_size)

        # save pickle for visualization
        results["intrinsics"] = data["intrinsic"][0].detach().cpu().numpy()
        with open(
            os.path.join(self.workspace, "validation", f"{self.name}.pkl"), "wb"
        ) as f:
            pickle.dump(results, f)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == "min" else -result
                )  # if max mode, use -result
            else:
                self.stats["results"].append(
                    average_loss
                )  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="val")
                metric.clear()

            if self.use_tensorboardX:
                self.writer.add_scalar("val/loss", average_loss, self.global_step)

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def get_results(self, preds, targets, results):
        results["images"].append(
            (preds["image"][0][..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
        )
        results["depths"].append(preds["depth"][0].detach().cpu().numpy())
        results["weights"].append(
            (preds["image"][0][..., 3].detach().cpu().numpy() * 255).astype(np.uint8)
        )
        results["images_gt"].append(
            (targets["image"][0][..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
        )
        if "depth" in targets:
            results["depths_gt"].append(targets["depth"][0].detach().cpu().numpy())
        results["poses"].append(targets["pose"][0].detach().cpu().numpy())
        return results

    def visualize(self, preds, targets):
        def put_text(img, txt):
            H, W, C = img.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            img[..., :3] = cv2.putText(
                np.array(img[..., :3]),
                txt,
                (10, H - 10),
                font,
                1,
                [255, 0, 0],
                2,
                cv2.LINE_AA,
            )
            return img

        # pred image
        pred_image = cv2.cvtColor(
            (preds["image"][0].detach().cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGBA2BGRA,
        )
        pred_image = put_text(pred_image, "pred_image")

        # pred depth
        pred_depth = distance2depth(
            preds["depth"][0].detach().cpu().numpy(),
            targets["intrinsic"][0].detach().cpu().numpy(),
        )
        pred_depth = cv2.applyColorMap(
            (pred_depth * 255 / self.depth_scale).astype(np.uint8), cv2.COLORMAP_JET
        ).astype(np.uint8)
        pred_depth = np.concatenate([pred_depth, pred_image[..., 3:]], axis=-1)
        pred_depth = cv2.cvtColor(pred_depth, cv2.COLOR_RGBA2BGRA)
        pred_depth = put_text(pred_depth, "pred_depth")

        frame = np.concatenate([pred_image, pred_depth], axis=0)

        # target image
        if "image" in targets:
            gt_image = (targets["image"][0].detach().cpu().numpy() * 255).astype(
                np.uint8
            )
            has_alpha = gt_image.shape[-1] == 4
            gt_image = cv2.cvtColor(
                gt_image, cv2.COLOR_RGBA2BGRA if has_alpha else cv2.COLOR_RGB2BGRA
            )
            gt_image = put_text(gt_image, "gt_image")

            # target depth
            if "depth" in targets:
                gt_depth = distance2depth(
                    targets["depth"][0].detach().cpu().numpy(),
                    targets["intrinsic"][0].detach().cpu().numpy(),
                )
                gt_depth = cv2.applyColorMap(
                    (gt_depth * 255 / self.depth_scale).astype(np.uint8),
                    cv2.COLORMAP_JET,
                ).astype(np.uint8)
                if has_alpha:
                    gt_depth = np.concatenate([gt_depth, gt_image[..., 3:]], axis=-1)
                    gt_depth = cv2.cvtColor(gt_depth, cv2.COLOR_RGBA2BGRA)
                else:
                    gt_depth = cv2.cvtColor(gt_depth, cv2.COLOR_RGB2BGRA)
                gt_depth = put_text(gt_depth, "gt_depth")

                frame = np.concatenate([gt_depth, frame], axis=0)
            frame = np.concatenate([gt_image, frame], axis=0)

        return frame

    def save_checkpoint(self, full=False, best=False):

        state = {
            "epoch": self.epoch,
            "stats": self.stats,
        }

        if self.model.cuda_ray:
            state["mean_count"] = self.model.mean_count
            state["mean_density"] = self.model.mean_density

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:

            state["model"] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if (
                    self.stats["best_result"] is None
                    or self.stats["results"][-1] < self.stats["best_result"]
                ):
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}"
                    )
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint."
                )

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(
                glob.glob(f"{self.ckpt_path}/{self.name}_ep*.pth.tar")
            )
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        if (not hasattr(self.model, "extrinsic_optimizer")) and (
            "extrinsic_optimizer.se3_refinement.weight" in checkpoint_dict["model"]
        ):
            del checkpoint_dict["model"]["extrinsic_optimizer.se3_refinement.weight"]

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and "ema" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema"])

        self.stats = checkpoint_dict["stats"]
        self.epoch = checkpoint_dict["epoch"]

        if self.model.cuda_ray:
            if "mean_count" in checkpoint_dict:
                self.model.mean_count = checkpoint_dict["mean_count"]
            if "mean_density" in checkpoint_dict:
                self.model.mean_density = checkpoint_dict["mean_density"]

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")

        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")

        if "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler, use default.")
