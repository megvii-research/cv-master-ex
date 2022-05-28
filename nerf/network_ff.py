import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding import get_encoder
from ffmlp import FFMLP
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from .renderer import NeRFRenderer

# class _trunc_exp(Function):
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float)
#     def forward(ctx, x):
#         ctx.save_for_backward(x)
#         return torch.exp(x)#.clamp(0, 1000)

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, g):
#         x = ctx.saved_tensors[0]
#         return g * torch.exp(x.clamp(-15, 15))

# trunc_exp = _trunc_exp.apply # why doesn't work?


class NeRFNetwork(NeRFRenderer):
    def __init__(
        self,
        encoding="hashgrid",
        encoding_dir="sphere_harmonics",
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        bound=1,
        cuda_ray=False,
        n_views=0,
        optimize_extrinsics=False,
        annealing_start=0.2,
        annealing_end=0.5,
        **kwargs
    ):
        super().__init__(bound, cuda_ray, n_views, optimize_extrinsics)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(
            encoding, desired_resolution=2048 * bound * 2
        )

        self.sigma_net = FFMLP(
            input_dim=self.in_dim,
            output_dim=1 + self.geo_feat_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir)
        self.in_dim_color += (
            self.geo_feat_dim + 1
        )  # a manual fixing to make it 32, as done in nerf_network.h#178

        self.color_net = FFMLP(
            input_dim=self.in_dim_color,
            output_dim=3,
            hidden_dim=self.hidden_dim_color,
            num_layers=self.num_layers_color,
        )

    def forward(self, x, d):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        d = d.view(-1, 3)

        # sigma
        x = self.encoder(x, bound=self.bound)
        h = self.sigma_net(x)

        # sigma = trunc_exp(h[..., 0])
        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)

        # TODO: avoid this cat op...
        # should pre-allocate output (col-major!), then inplace write from ffmlp & shencoder, finally transpose to row-major.
        # this is impossible...
        p = torch.zeros_like(geo_feat[..., :1])  # manual input padding
        h = torch.cat([d, geo_feat, p], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        sigma = sigma.view(*prefix)
        color = color.view(*prefix, -1)

        return sigma, color

    def density(self, x):
        # x: [B, N, 3], in [-bound, bound]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)

        x = self.encoder(x, bound=self.bound)
        h = self.sigma_net(x)

        # sigma = trunc_exp(h[..., 0])
        sigma = F.relu(h[..., 0])
        sigma = sigma.view(*prefix)

        return sigma
