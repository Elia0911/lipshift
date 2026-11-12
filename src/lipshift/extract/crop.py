"""Face keypoints with the MediaPipe Tasks FaceDetector, mouth crops with Auto-AVSR.

Auto-AVSR's own detector uses ``mediapipe.solutions``, which is absent from mediapipe
0.10.30+ and 1.x. ``TasksLandmarksDetector`` is a drop-in replacement: BlazeFace returns
six keypoints in the same order as the legacy API (right eye, left eye, nose tip, mouth
center, right ear, left ear) and Auto-AVSR uses the first four. On a test image the two
APIs gave identical keypoints to three decimals (short-range model).

Differences from the original: only the short-range model is used (the legacy code tried
the full-range model first) and the largest face is selected by bounding-box area.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lipshift.extract.repos import switch_repo


class TasksLandmarksDetector:
    """Per-frame four-point face landmarks in pixel coordinates."""

    def __init__(self, model_path: str | Path, min_confidence: float = 0.5) -> None:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        options = vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def close(self) -> None:
        """Release the native detector."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self) -> TasksLandmarksDetector:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __call__(self, video_frames: np.ndarray) -> list[np.ndarray | None]:
        """Return one (4, 2) int array per frame, or None where no face was found.

        Raises:
            AssertionError: if no frame has a face (same behaviour as Auto-AVSR).
        """
        import mediapipe as mp

        landmarks: list[np.ndarray | None] = []
        for frame in video_frames:
            h, w = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            detections = self._detector.detect(image).detections
            if not detections:
                landmarks.append(None)
                continue
            best = max(detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            landmarks.append(np.array([[int(k.x * w), int(k.y * h)] for k in best.keypoints[:4]]))
        if all(lm is None for lm in landmarks):
            raise AssertionError("Cannot detect any frames in the video")
        return landmarks


def crop_mouths(
    frames_rgb: np.ndarray, auto_avsr_dir: str | Path, face_model_path: str | Path
) -> np.ndarray:
    """Detect faces and return aligned RGB mouth crops of shape (T, 96, 96, 3)."""
    switch_repo(auto_avsr_dir)
    from preparation.detectors.mediapipe.video_process import VideoProcess

    with TasksLandmarksDetector(face_model_path) as detector:
        landmarks = detector(frames_rgb)
    mouths = VideoProcess(convert_gray=False)(frames_rgb, landmarks)
    if mouths is None:
        raise RuntimeError("Mouth cropping returned None (no usable landmarks)")
    return np.asarray(mouths)
