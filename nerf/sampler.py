import importlib
from typing import Dict

import torch
import torch.nn.functional as F


class Sampler(object):
    def __init__(self, sampler: str, **kwargs):
        self.ray_sampler = self.get_sampler(sampler)(**kwargs)

    def get_sampler(self, sampler_name: str):
        return getattr(importlib.import_module(f".", package=__name__), sampler_name)

    def sample(self, full_grid=False, **data):
        return self.ray_sampler.sample(full_grid=full_grid, **data)


class GridRaySampler(object):
    def __init__(self, num_rays: int, **kwargs):
        self.num_rays = num_rays

    def sample(
        self,
        pose: torch.Tensor,
        intrinsic: torch.Tensor,
        H: int,
        W: int,
        full_grid: bool = False,
        **kwargs,
    ):
        """
        Args:
            pose: [B, 4, 4]
            intrinsic: [B, 3, 3]

        Returns:
            rays_o, rays_d: [B, N_rays, 3]
            select_inds: [B, N_rays]
        """
        H, W = int(H[0]), int(W[0])
        N_rays = self.num_rays if not full_grid else H * W
        device = pose.device
        rays_o = pose[..., :3, 3]  # [B, 3]
        prefix = pose.shape[:-2]

        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W, device=device),
            torch.linspace(0, H - 1, H, device=device),
            indexing="ij",
        )  # for torch < 1.10, should remove indexing='ij'
        i = i.t().reshape([*[1] * len(prefix), H * W]).expand([*prefix, H * W])
        j = j.t().reshape([*[1] * len(prefix), H * W]).expand([*prefix, H * W])

        rays_cnt = {}
        if not full_grid:
            select_inds = torch.randint(
                0,
                len(
                    i.view(
                        -1,
                    )
                ),
                size=[N_rays],
                device=device,
            )
            select_inds = select_inds.expand([*prefix, N_rays])
            i = torch.gather(i, -1, select_inds)
            j = torch.gather(j, -1, select_inds)
        else:
            select_inds = torch.arange(N_rays, device=device).expand([*prefix, N_rays])

        pixel_points_cam = lift(i, j, torch.ones_like(i), intrinsics=intrinsic)
        pixel_points_cam = pixel_points_cam.transpose(-1, -2)

        world_coords = torch.bmm(pose, pixel_points_cam).transpose(-1, -2)[..., :3]

        rays_d = world_coords - rays_o[..., None, :]
        rays_d = F.normalize(rays_d, dim=-1)

        rays_o = rays_o[..., None, :].expand_as(rays_d)

        return rays_o, rays_d, select_inds


class SemanticRaySampler:
    def __init__(
        self,
        num_rays: int,
        class_colors: Dict,
        mask_away_instance: bool = False,
        extra_sampling: bool = False,
        class_ratio: Dict = None,
        **kwargs,
    ):
        self.num_rays = num_rays
        self.class_colors = class_colors
        self.mask_away_instance = mask_away_instance
        self.extra_sampling = extra_sampling
        self.class_ratio = class_ratio

    def sample(
        self,
        pose: torch.Tensor,
        intrinsic: torch.Tensor,
        instance_seg: torch.Tensor,
        class_seg: torch.Tensor,
        H: int,
        W: int,
        full_grid: bool = False,
        **kwargs,
    ):
        """
        Args:
            pose: [B, 4, 4]
            intrinsic: [B, 3, 3]
            instance_seg: [B, H, W, 3]
            class_seg: [B, H, W, 3]

        Returns:
            rays_o, rays_d: [B, N_rays, 3]
            select_inds: [B, N_rays]
        """
        H, W = int(H[0]), int(W[0])
        N_rays = self.num_rays if not full_grid else H * W
        device = pose.device
        rays_o = pose[..., :3, 3]  # [B, 3]
        prefix = pose.shape[:-2]

        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W, device=device),
            torch.linspace(0, H - 1, H, device=device),
            indexing="ij",
        )  # for torch < 1.10, should remove indexing='ij'
        i = i.t().reshape([*[1] * len(prefix), H * W]).expand([*prefix, H * W])
        j = j.t().reshape([*[1] * len(prefix), H * W]).expand([*prefix, H * W])

        rays_cnt = {}
        if not full_grid:
            if self.mask_away_instance:
                kept_mask = (instance_seg.to(torch.float32).sum(axis=3) == 0).view(
                    *prefix, H * W
                )  # 有效的区域是1
                # kept_mask # 去掉动态物体
                N_rays = min(N_rays, kept_mask.sum())
            else:
                kept_mask = torch.ones_like(i)
            remain_rays = N_rays  # 还需要采多少rays
            inds = []
            i_extra = []  # contains classes with extra ray sample and kept_mask sample
            j_extra = []  # contains classes with extra ray sample and kept_mask sample
            if self.extra_sampling:
                for class_ in self.class_ratio:
                    if self.class_ratio[class_] > 0:
                        ray_num_extra = int(
                            self.class_ratio[class_] * N_rays
                        )  # ray_num_extra
                        cb, cg, cr = self.class_colors[class_]
                        class_mask = (
                            (class_seg.view(-1, 3)[:, 0] == cb)
                            * (class_seg.view(-1, 3)[:, 1] == cg)
                            * (class_seg.view(-1, 3)[:, 2] == cr)
                        ).view(*prefix, H * W)
                        # bg_mask -= class_mask.int() 背景采样的时候要不要去掉之前采样过的种类？
                        i_mask = i[class_mask].view(*prefix, -1)
                        j_mask = j[class_mask].view(*prefix, -1)
                        if (
                            len(
                                i_mask.view(
                                    -1,
                                )
                            )
                            > 0
                        ):
                            rays_cnt[class_] = [
                                N_rays - remain_rays,
                                N_rays - remain_rays + ray_num_extra,
                            ]  # [start, end] rays inds in this class
                            remain_rays -= ray_num_extra
                            if remain_rays < 0:
                                print(
                                    "Please set the sum of sample ratio  smaller than 1"
                                )
                                break
                            select_inds = torch.randint(
                                0,
                                len(
                                    i_mask.view(
                                        -1,
                                    )
                                ),
                                size=[ray_num_extra],
                                device=device,
                            )
                            select_inds = select_inds.expand([*prefix, ray_num_extra])
                            i_extra_class = torch.gather(i_mask, -1, select_inds)
                            j_extra_class = torch.gather(j_mask, -1, select_inds)
                            i_extra.append(i_extra_class)
                            j_extra.append(j_extra_class)
                            select_inds_global = (
                                i_extra_class + j_extra_class * W
                            )  # 计算全局index
                            inds.append(select_inds_global.long())
            if remain_rays > 0:
                # 如果mask_away_instance和extra_sampling都不执行，remain_rays=N_rays，正常对有效区域采样
                i_remian_mask = i[kept_mask.bool()].view(*prefix, -1)
                j_remian_mask = j[kept_mask.bool()].view(*prefix, -1)
                select_inds = torch.randint(
                    0,
                    len(
                        i_remian_mask.view(
                            -1,
                        )
                    ),
                    size=[remain_rays],
                    device=device,
                )
                select_inds = select_inds.expand([*prefix, remain_rays])
                rays_cnt["remain"] = [N_rays - remain_rays, N_rays]
                i_remain_class = torch.gather(i_remian_mask, -1, select_inds)
                j_remain_class = torch.gather(j_remian_mask, -1, select_inds)
                i_extra.append(i_remain_class)
                j_extra.append(j_remain_class)
                select_inds_global = i_remain_class + j_remain_class * W  # 计算全局index
                inds.append(select_inds_global.long())

            i = torch.cat(i_extra, dim=1)
            j = torch.cat(j_extra, dim=1)
            select_inds = torch.cat(inds, dim=1)
        else:
            select_inds = torch.arange(N_rays, device=device).expand([*prefix, N_rays])

        pixel_points_cam = lift(i, j, torch.ones_like(i), intrinsics=intrinsic)
        pixel_points_cam = pixel_points_cam.transpose(-1, -2)

        world_coords = torch.bmm(pose, pixel_points_cam).transpose(-1, -2)[..., :3]

        rays_d = world_coords - rays_o[..., None, :]
        rays_d = F.normalize(rays_d, dim=-1)

        rays_o = rays_o[..., None, :].expand_as(rays_d)

        return rays_o, rays_d, select_inds


def lift(x, y, z, intrinsics):
    # x, y, z: [B, N]
    # intrinsics: [B, 3, 3]

    fx = intrinsics[..., 0, 0].unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1)
    sk = intrinsics[..., 0, 1].unsqueeze(-1)

    x_lift = (x - cx + cy * sk / fy - sk * y / fy) / fx * z
    y_lift = (y - cy) / fy * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z)), dim=-1)
