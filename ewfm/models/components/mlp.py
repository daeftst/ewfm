import copy

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0]).to(x.device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size).to(x.device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class Block(nn.Module):
    def __init__(self, size: int, t_emb_size: int = 0, add_t_emb=False, concat_t_emb=False):
        super().__init__()

        in_size = size + t_emb_size if concat_t_emb else size
        self.ff = nn.Linear(in_size, size)
        self.act = nn.GELU()

        self.add_t_emb = add_t_emb
        self.concat_t_emb = concat_t_emb

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        in_arg = torch.cat([x, t_emb], dim=-1) if self.concat_t_emb else x
        out = x + self.act(self.ff(in_arg))

        if self.add_t_emb:
            out = out + t_emb

        return out


class MyMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        out_dim: int = 2,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
        add_t_emb: bool = False,
        concat_t_emb: bool = False,
        input_dim: int = 2,
        energy_function=None,
    ):
        super().__init__()

        self.add_t_emb = add_t_emb
        self.concat_t_emb = concat_t_emb

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)

        positional_embeddings = []
        for i in range(input_dim):
            embedding = PositionalEmbedding(emb_size, input_emb, scale=25.0)

            self.add_module(f"input_mlp{i}", embedding)

            positional_embeddings.append(embedding)

        self.channels = 1
        self.self_condition = False
        concat_size = len(self.time_mlp.layer) + sum(
            map(lambda x: len(x.layer), positional_embeddings)
        )

        layers = [nn.Linear(concat_size, hidden_size)]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size, emb_size, add_t_emb, concat_t_emb))

        in_size = emb_size + hidden_size if concat_t_emb else emb_size
        layers.append(nn.Linear(in_size, out_dim))

        self.layers = layers
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, t, x, x_self_cond=False):
        positional_embs = [
            self.get_submodule(f"input_mlp{i}")(x[:, i]) for i in range(x.shape[-1])
        ]

        if t.ndim == 0:  # scalar t
            t = t.expand(x.shape[0])  # Expand t to match batch size of x
        t_emb = self.time_mlp(t)

        x = torch.cat((*positional_embs, t_emb), dim=-1)

        for i, layer in enumerate(self.layers):
            if i == 0:
                x = nn.GELU()(layer(x))
                if self.add_t_emb:
                    x = x + t_emb

            elif i == len(self.layers) - 1:
                if self.concat_t_emb:
                    x = torch.cat([x, t_emb], dim=-1)

                x = layer(x)

            else:
                x = layer(x, t_emb)

        return x
