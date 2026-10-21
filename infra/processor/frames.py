"""Sample ~1 frame per second from a clip with OpenCV."""

from pathlib import Path

import cv2
import numpy as np

EVERY_S = 1.0
MAX_FRAMES = 30


def sample(path: Path, every_s: float = EVERY_S, max_frames: int = MAX_FRAMES) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 0 < fps <= 240 else 30.0
    out, idx, next_t = [], 0, 0.0
    try:
        while len(out) < max_frames and cap.grab():
            # Timestamp from the container when it has one (phones record variable frame rate), else index / fps.
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = ms / 1000 if ms > 0 or idx == 0 else idx / fps
            if t + 1e-6 >= next_t:
                ok, frame = cap.retrieve()
                if ok:
                    out.append(frame)
                    next_t += every_s
            idx += 1
    finally:
        cap.release()
    if not out:
        raise ValueError(f"no frames decoded from {path.name}")
    return out
