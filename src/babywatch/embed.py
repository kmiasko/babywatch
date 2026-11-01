"""Frozen backbone feature extractor (DINOv2 / SigLIP2 via timm), batched, fp16 on CUDA."""
from __future__ import annotations

import numpy as np
import timm
import torch

BACKBONES = {
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
    "siglip2_b": "vit_base_patch16_siglip_224.v2_webli",
}


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, backbone: str = "dinov2_b", img_size: int = 224, device: str | None = None):
        self.backbone, self.img_size = backbone, img_size
        self.device = device or pick_device()
        name = BACKBONES[backbone]
        kw = {"img_size": img_size} if backbone.startswith("dinov2") else {}
        self.model = timm.create_model(name, pretrained=True, num_classes=0, **kw).eval().to(self.device)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.device == "cuda":
            self.model = self.model.half()
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=self.device).view(1, 3, 1, 1)
        self.size = img_size if backbone.startswith("dinov2") else cfg["input_size"][-1]

    @torch.inference_mode()
    def __call__(self, crops_bgr: list[np.ndarray], batch: int = 64) -> np.ndarray:
        """crops_bgr: list of HxWx3 uint8 BGR (already self.size square). Returns (N, D) float32."""
        out = []
        for i in range(0, len(crops_bgr), batch):
            x = torch.from_numpy(np.stack(crops_bgr[i : i + batch])[..., ::-1].copy())
            x = x.to(self.device).permute(0, 3, 1, 2).float().div_(255)
            if x.shape[-1] != self.size:
                x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True)
            x = ((x - self.mean) / self.std).to(self.dtype)
            out.append(self._features(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.model.forward_features(x)
        if self.backbone.startswith("dinov2"):
            # standard DINOv2 linear-probe feature: [CLS, mean(patch tokens)]
            n_prefix = getattr(self.model, "num_prefix_tokens", 1)
            return torch.cat([tok[:, 0], tok[:, n_prefix:].mean(1)], dim=1)
        return self.model.forward_head(tok, pre_logits=True)  # SigLIP attention-pool
