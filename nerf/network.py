import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding import get_encoder

from .renderer import NeRFRenderer


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
        annealing_start_level=0,
        **kwargs
    ):
        super().__init__(bound, cuda_ray, n_views, optimize_extrinsics)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(
            encoding,
            desired_resolution=2048 * bound * 2,
            annealing_start=annealing_start,
            annealing_end=annealing_end,
            annealing_start_level=annealing_start_level,
        )

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir)
        self.in_dim_color += self.geo_feat_dim

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_color
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

    def forward(self, x, d, progress):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound, progress=progress)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        # color

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [B, N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        # sigma = torch.exp(torch.clamp(h[..., 0], -15, 15))
        sigma = F.relu(h[..., 0])

        return sigma
