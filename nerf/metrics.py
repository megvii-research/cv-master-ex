import importlib
import os
from typing import List

import lpips
import torch


def get_metrics(metric_names: List):
    metric_list = []
    for metric_name in metric_names:
        metric_name = metric_name + "Meter"
        metric_list.append(
            getattr(importlib.import_module(f".", package=__name__), metric_name)()
        )
    return metric_list


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.clear()

    def clear(self):
        self.val = 0
        self.avg = 0
        self.min = 1e10
        self.max = -1e10
        self.sum = 0
        self.count = 0

    def _update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.min = min(self.min, val)
        self.max = max(self.max, val)


class MetricMeter(AverageMeter):
    def __init__(self, name):
        super().__init__()
        self.name = name

        self.clear()

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach()
            outputs.append(inp)

        return outputs

    def calculate(self, preds, truths):
        raise NotImplementedError()

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)

        val, n = self.calculate(preds, truths)

        self._update(val, n)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, self.name), self.avg, global_step)

    def __str__(self):
        return f"{self.name} = {self.val:.4f}({self.avg:.4f}) "

    def report(self):
        return (
            f"{self.name}: avg({self.avg:.4f}) min({self.min:.4f}) max({self.max:.4f})"
        )


class PSNRMeter(MetricMeter):
    def __init__(self):
        super().__init__("PSNR")

    def prepare_inputs(self, *inputs):
        outputs = []
        for inp in inputs:
            assert (
                "image" in inp
            ), f"{self.name} cannot be calculated due to missing relevant inputs!"
            outputs.append(inp["image"].detach()[..., :3])

        return outputs

    def calculate(self, preds: torch.Tensor, truths: torch.Tensor):
        dim = tuple(range(1, preds.ndim))
        mse = torch.pow(preds.double() - truths.view_as(preds).double(), 2).mean(
            dim=dim
        )
        psnr = torch.sum(10.0 * torch.log10(1.0 / (mse + 1e-10))) / preds.shape[0]
        n = preds.shape[0]
        return psnr.item(), n


class LPIPSMeter(MetricMeter, torch.nn.Module):
    """
    Perceptual loss.
    """

    def __init__(self):
        MetricMeter.__init__(self, "LPIPS")
        torch.nn.Module.__init__(self)

        self.lpips = lpips.LPIPS(net="vgg")

    def prepare_inputs(self, *inputs):
        outputs = []
        for inp in inputs:
            assert (
                "image" in inp
            ), f"{self.name} cannot be calculated due to missing relevant inputs!"
            image = inp["image"][..., :3]
            if len(image.shape) == 4:
                assert image.shape[0] == 1
                image = image.permute(0, 3, 1, 2)
                outputs.append(image.detach())
            else:
                outputs.append([])

        return outputs

    def calculate(self, preds: torch.Tensor, truths: torch.Tensor):
        """
        The preds and truths should be torch RGB image in shape [1, 3, h, w], normalized to [-1,1].
        """
        if len(preds) == 0 or len(truths) == 0:
            return 0, 1
        return self.lpips(preds, truths).item(), 1
