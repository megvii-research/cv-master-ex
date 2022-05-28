import importlib
import warnings
from typing import Dict

import torch


class Criterion(torch.nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()

        self.loss_funcs = torch.nn.ModuleDict()
        for loss_name, weight in cfg.items():
            if weight > 0:
                self.loss_funcs[loss_name] = self.get_loss(loss_name)(weight)

    def get_loss(self, loss_name):
        loss_name_capital = "".join(
            [name.capitalize() for name in loss_name.split("_")]
        )
        return getattr(
            importlib.import_module(f".", package=__name__), loss_name_capital
        )

    def forward(self, preds: Dict, targets: Dict):
        losses = {}
        for loss_name, loss_func in self.loss_funcs.items():
            losses[loss_name] = loss_func(preds, targets)
        return losses


class ImageLoss(torch.nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.loss = torch.nn.HuberLoss(delta=0.1)
        self.weight = weight

    def forward(self, preds: Dict, targets: Dict) -> torch.Tensor:
        if "image" not in preds or "image" not in targets:
            warnings.warn(
                "Image loss cannot be calculated due to missing relevant inputs!"
            )
            return 0

        pred = preds["image"]
        target = targets["image"]

        return self.loss(pred[..., :3], target[..., :3]) * self.weight


class DepthLoss(torch.nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        self.weight = weight

    def forward(self, preds: Dict, targets: Dict) -> torch.Tensor:
        if "depth" not in preds or "depth" not in targets:
            warnings.warn(
                "Depth loss cannot be calculated due to missing relevant inputs!"
            )
            return 0

        pred = preds["depth"]
        target = targets["depth"]
        mask = target > 0
        return (
            self.loss(pred[mask], target[mask])
            / (target.max() - target.min())
            * self.weight
        )


class TransmittanceLoss(torch.nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        self.weight = weight

    def forward(self, preds: Dict, targets: Dict) -> torch.Tensor:
        if (
            "image" not in preds
            or "image" not in targets
            or preds["image"].shape[-1] != 4
            or targets["image"].shape[-1] != 4
        ):
            warnings.warn(
                "Transmittance loss cannot be calculated due to missing relevant inputs!"
            )
            return 0

        pred = preds["image"][..., 3:]
        target = targets["image"][..., 3:]
        return self.loss(pred, target) * self.weight


class DepthVarianceRegularization(torch.nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, preds: Dict, targets: Dict) -> torch.Tensor:
        if "depth_variance" not in preds:
            warnings.warn(
                "Depth variance Regularization cannot be calculated due to missing relevant inputs!"
            )
            return 0

        pred = preds["depth_variance"]
        if "image" in targets and targets["image"].shape[-1] == 4:
            mask = targets["image"][..., 3]
        else:
            mask = preds["image"][..., 3].detach()
        return torch.mean(pred * mask) * self.weight
