"""Tests for encoder preprocessing that do not need upstream repositories."""

import numpy as np
import pytest
import torch

pytest.importorskip("torchvision")
pytest.importorskip("cv2")

from lipshift.extract.encoders import MultilingualVSREncoder  # noqa: E402


def test_multilingual_preprocess_shape_and_normalization() -> None:
    mouths = np.full((7, 96, 96, 3), 128, np.uint8)
    x = MultilingualVSREncoder.preprocess(mouths)
    assert x.shape == (1, 7, 88, 88)
    assert x.dtype == torch.float32
    expected = (128 / 255.0 - 0.421) / 0.165
    assert torch.allclose(x, torch.full_like(x, expected), atol=1e-5)
