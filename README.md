# lipshift

Where lip-reading deepfake detectors break: language shift, generator shift and calibrated abstention.

**Status: work in progress.** No results yet. This README will be completed with real numbers once the experiments have run.

## What this project studies

Talking-face deepfake detectors built on frozen visual speech recognition (VSR) features are a promising direction (see FauxNet, Bora et al., arXiv 2511.22443). This repository tests where that approach fails:

- **Study A, language shift.** Lip-reading encoders are trained on specific languages. We compare an English and a Spanish encoder with the same architecture on English and Spanish clips.
- **Study B, generator shift.** We test whether calibration and split-conformal coverage hold when a generator unseen in training appears at test time, and whether abstention and score fusion restore usable guarantees.

## Current state

- Feature extraction pipeline: video decoding, 25 fps normalization, MediaPipe Tasks face keypoints, Auto-AVSR mouth alignment, frozen encoders (Auto-AVSR English; multilingual English and Spanish).
- Resumable batch extraction for Colab, with atomic writes.
- Stratified subset selection over MAVOS-DD metadata.

## Quickstart (development)

```bash
git clone https://github.com/<your-user>/lipshift.git
cd lipshift
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not integration"
```

## Data and model licenses

This repository contains code only (MIT). It does not redistribute videos, mouth crops, extracted features or model weights.

| Asset | Terms |
|---|---|
| MAVOS-DD (Croitoru et al., 2025) | CC BY-NC-SA 4.0, gated access on Hugging Face |
| Auto-AVSR code (Ma et al., 2023) | Apache-2.0; pretrained models: see the upstream model zoo terms |
| Multilingual VSR models (Ma et al., 2022) | Non-commercial research and benchmarking use |
| MediaPipe BlazeFace model | See the MediaPipe model card |

## Ethics

This is detection research. The project does not generate or publish synthetic media of identifiable people.

## Contact

Eli OUN, [github.com/Elia0911](https://github.com/Elia0911)
