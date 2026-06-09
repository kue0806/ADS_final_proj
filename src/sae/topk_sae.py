"""Top-k sparse autoencoder."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, dict_size: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k

        self.encoder = nn.Linear(d_model, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, d_model, bias=True)

        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        with torch.no_grad():
            W = torch.randn(d_model, dict_size)
            W /= W.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.decoder.weight.copy_(W)
            self.encoder.weight.copy_(W.t())

    @torch.no_grad()
    def init_bias_from_mean(self, x_mean: torch.Tensor):
        self.decoder.bias.copy_(x_mean.to(self.decoder.bias.device))

    @torch.no_grad()
    def renorm_decoder(self):
        """Renormalize decoder columns to unit L2 (call after each step)."""
        W = self.decoder.weight
        W.div_(W.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d_model) -> (B, dict_size) with exactly k nonzeros per row."""
        z = F.relu(self.encoder(x - self.decoder.bias))
        vals, idx = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter_(-1, idx, vals)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes)

    def forward(self, x: torch.Tensor):
        codes = self.encode(x)
        return self.decode(codes), codes
