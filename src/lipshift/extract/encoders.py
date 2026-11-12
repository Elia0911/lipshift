"""Frozen visual speech recognition (VSR) encoders.

- ``AutoAVSREncoder``: Auto-AVSR conformer (Ma et al., ICASSP 2023), 768-d features.
  Code: https://github.com/mpc001/auto_avsr (Apache-2.0). Weights: model zoo terms.
- ``MultilingualVSREncoder``: models from "Visual Speech Recognition for Multiple
  Languages in the Wild" (Ma et al., Nature Machine Intelligence 2022), built from the
  ``model.json`` shipped with each checkpoint. Code and models: non-commercial use.

Both expect mouth crops at 25 fps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from lipshift.extract.repos import switch_repo


def _load_state(ckpt_path: str | Path) -> dict:
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def _report(result) -> dict[str, list[str]]:
    return {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}


class AutoAVSREncoder(torch.nn.Module):
    """Auto-AVSR visual front-end + conformer encoder, decoder unused."""

    feature_dim = 768

    def __init__(self, repo_dir: str | Path) -> None:
        super().__init__()
        self.repo_dir = Path(repo_dir)
        switch_repo(self.repo_dir)
        from datamodule.transforms import TextTransform
        from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

        self.model = E2E(len(TextTransform().token_list), "video", ctc_weight=0.1)

    def load_checkpoint(self, ckpt_path: str | Path) -> dict[str, list[str]]:
        """Load weights non-strictly and return missing / unexpected keys."""
        return _report(self.model.load_state_dict(_load_state(ckpt_path), strict=False))

    def preprocess(self, mouths_rgb: np.ndarray) -> torch.Tensor:
        """Official test transform: /255, center crop 88, grayscale, normalize -> (T,1,88,88)."""
        switch_repo(self.repo_dir)
        from datamodule.transforms import VideoTransform

        return VideoTransform("test")(torch.from_numpy(mouths_rgb).permute(0, 3, 1, 2))

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(T, 1, 88, 88) -> (T, 768)."""
        self.model.eval()
        y = self.model.frontend(x.unsqueeze(0))
        y = self.model.proj_encoder(y)
        y, _ = self.model.encoder(y, None)
        return y.squeeze(0)


class MultilingualVSREncoder(torch.nn.Module):
    """Encoder of a multilingual-model-zoo checkpoint, built from its ``model.json``."""

    def __init__(
        self,
        repo_dir: str | Path,
        model_conf: str | Path | None = None,
        args: argparse.Namespace | None = None,
    ) -> None:
        super().__init__()
        switch_repo(repo_dir)
        from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

        if args is None:
            if model_conf is None:
                raise ValueError("Give model_conf (path to model.json) or args")
            with open(model_conf, "rb") as f:
                confs = json.load(f)
            args = argparse.Namespace(**(confs if isinstance(confs, dict) else confs[2]))
            if getattr(args, "labels_type", "char") in (None, "char"):
                odim = len(args.char_list)
            else:  # unigram5000: <blank> + units + <eos>, as in pipelines/model.py
                tok = Path(repo_dir) / "pipelines" / "tokens" / "unigram5000_units.txt"
                odim = len(tok.read_text().splitlines()) + 2
        else:
            odim = 10  # test path: stand-in config, untrained weights
        self.args = args
        self.feature_dim = int(args.adim)
        self.model = E2E(odim, args)

    def load_checkpoint(self, ckpt_path: str | Path) -> dict[str, list[str]]:
        """Load weights non-strictly and return missing / unexpected keys."""
        return _report(self.model.load_state_dict(_load_state(ckpt_path), strict=False))

    @staticmethod
    def preprocess(mouths_rgb: np.ndarray) -> torch.Tensor:
        """Gray, /255, center crop 88, normalize -> (1, T, 88, 88).

        Upstream converts to gray before the affine warp; we convert the warped RGB crop.
        Both steps are linear, so the difference is limited to interpolation rounding.
        """
        import torchvision

        gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in mouths_rgb])
        x = torch.from_numpy(gray).float().unsqueeze(0) / 255.0
        x = torchvision.transforms.CenterCrop(88)(x)
        return torchvision.transforms.Normalize(0.421, 0.165)(x)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(1, T, 88, 88) -> (T, adim)."""
        self.model.eval()
        return self.model.encode(x)
