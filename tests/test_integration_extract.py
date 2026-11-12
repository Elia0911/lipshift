"""End-to-end extraction with untrained weights.

Skipped unless LIPSHIFT_AUTO_AVSR (cloned repo) and LIPSHIFT_FACE_MODEL (BlazeFace
.tflite) are set. Uses the public-domain scikit-image astronaut photo as a face.
"""

import os
from pathlib import Path

import numpy as np
import pytest

AUTO_AVSR = os.environ.get("LIPSHIFT_AUTO_AVSR")
FACE_MODEL = os.environ.get("LIPSHIFT_FACE_MODEL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (AUTO_AVSR and FACE_MODEL), reason="upstream repo / face model not set"),
]


@pytest.fixture(scope="module")
def face_clip(tmp_path_factory) -> Path:
    cv2 = pytest.importorskip("cv2")
    data = pytest.importorskip("skimage.data")
    img = data.astronaut()
    path = tmp_path_factory.mktemp("clip") / "astro30.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (512, 512))
    for t in range(60):
        m = np.float32([[1, 0, t * 0.3], [0, 1, 0]])
        w.write(cv2.cvtColor(cv2.warpAffine(img, m, (512, 512)), cv2.COLOR_RGB2BGR))
    w.release()
    return path


def test_auto_avsr_pipeline_on_face_clip(face_clip: Path) -> None:
    from lipshift.extract.encoders import AutoAVSREncoder
    from lipshift.extract.pipeline import extract_clip

    enc = AutoAVSREncoder(AUTO_AVSR)
    feats, info = extract_clip(face_clip, enc, AUTO_AVSR, FACE_MODEL)
    assert feats.shape == (50, 768)  # 60 frames at 30 fps -> 50 at 25 fps
    assert np.isfinite(feats).all()
    assert info["n_frames_25fps"] == 50
