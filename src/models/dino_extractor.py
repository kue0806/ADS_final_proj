"""Frozen ViT feature extractor (HuggingFace AutoModel)."""
from __future__ import annotations
import torch
from torch import nn
from transformers import AutoModel


class DinoExtractor(nn.Module):
    """Frozen patch-token feature extractor.

    model_name: an HF model id (e.g. 'facebook/dinov2-base'), or
                'random:<id>' to build the architecture with random weights.
    layer:      hidden-state index (negative = from the end).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        layer: int = -2,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self.dtype = dtype

        # 'random:<id>' builds the same architecture with random weights
        # (fixed seed) as a no-pretraining baseline.
        self.random_init = model_name.startswith("random:")
        if self.random_init:
            from transformers import AutoConfig
            base_name = model_name.split("random:", 1)[1]
            torch.manual_seed(0)
            config = AutoConfig.from_pretrained(base_name)
            self.model = AutoModel.from_config(config).to(device, dtype=dtype).eval()
        else:
            try:
                self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device).eval()
            except TypeError:
                self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = self.model.config
        self.hidden_dim: int = cfg.hidden_size
        self.patch_size: int = getattr(cfg, "patch_size", 14)
        self.num_layers: int = getattr(cfg, "num_hidden_layers", -1)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, 3, H, W) -> {'cls': (B, D), 'patches': (B, N, D), 'h': (B, 1+N, D)}."""
        x = x.to(self.device, dtype=self.dtype, non_blocking=True)
        out = self.model(x, output_hidden_states=True)
        h = out.hidden_states[self.layer]  # (B, 1+N, D)
        return {
            "cls": h[:, 0, :].contiguous(),
            "patches": h[:, 1:, :].contiguous(),
            "h": h,
        }

    def grid_size(self, input_hw: int = 224) -> int:
        """Patch grid side length for a square input."""
        return input_hw // self.patch_size
