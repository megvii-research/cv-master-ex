import torch
import torch.nn as nn
import torch.nn.functional as F

from hashencoder import AnnealableHashEncoder, HashEncoder
from shencoder import SHEncoder


class FreqEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        max_freq_log2,
        N_freqs,
        log_sampling=True,
        include_input=True,
        periodic_fns=(torch.sin, torch.cos),
    ):

        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)

        return out


def get_encoder(
    encoding,
    input_dim=3,
    multires=6,
    degree=4,
    num_levels=16,
    level_dim=2,
    base_resolution=16,
    log2_hashmap_size=19,
    desired_resolution=2048,
    annealing_start=0.2,
    annealing_end=0.5,
    annealing_start_level=0,
    **kwargs
):

    if encoding == "None":
        return lambda x, **kwargs: x, input_dim

    elif encoding == "frequency":
        encoder = FreqEncoder(
            input_dim=input_dim,
            max_freq_log2=multires - 1,
            N_freqs=multires,
            log_sampling=True,
        )

    elif encoding == "sphere_harmonics":
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == "hashgrid":
        encoder = HashEncoder(
            input_dim=input_dim,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution,
        )

    elif encoding == "annealable_hashgrid":
        encoder = AnnealableHashEncoder(
            input_dim=input_dim,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution,
            annealing_start=annealing_start,
            annealing_end=annealing_end,
            annealing_start_level=annealing_start_level,
        )

    else:
        raise NotImplementedError()

    return encoder, encoder.output_dim
