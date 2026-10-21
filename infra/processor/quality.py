"""Quality gate before the cause models: per sampled frame, reject "dark", "blur" or "no_sky".

Features (each frame first downscaled to WIDTH px wide, so thresholds don't depend on the phone's resolution):
- luminance: mean of Y = 0.299 R + 0.587 G + 0.114 B (0-255). Night / covered lens -> "dark".
- sharpness: variance of the Laplacian of the grey frame, per tile on a TILES x TILES grid, 80th percentile over
  tiles. A plain sky has little texture, so whole-frame variance called sharp sky shots "blurred"; a really blurred
  frame is soft in every tile, a sharp sky shot still has a sharp horizon or building somewhere. -> "blur".
- sky fraction: share of pixels in the top SKY_TOP of the frame that look like sky: bright and low-saturation
  (overcast, haze, white sky) or blue hue. Indoor / camera pointed at the ground -> "no_sky".
A clip needs MIN_PASS passing frames, else the cause models are skipped.

Calibration (10 Oct 2026, ~/datasets/own_clips, 24 clips, ~1 frame/s as the processor samples; grid search over
the sky-pixel definition and the three thresholds): 21 of 216 settings keep every outdoor clip at >= 3 passing frames
and reject reject_indoor_01; chosen = fewest outdoor frame rejects, sharpness in the middle of the feasible 8-11 band.
Margins are thin: reject_indoor_01 keeps 2 frames (needs < 3) and fire_smoke_01 (smoky, little open sky) passes 9/15.
The only outdoor frames rejected are 2 black fade-in frames ("dark"), 1 motion-blurred frame and 6 low-sky frames of
fire_smoke_01. Recalibrate when more own clips arrive.
"""

import cv2
import numpy as np

WIDTH = 320
TILES, TILE_PERCENTILE = 4, 80
SKY_TOP = 0.40
MIN_PASS = 3
REASONS = ("dark", "blur", "no_sky")

# Sky-like pixel in OpenCV HSV (H 0-179, S and V 0-255).
SKY_BRIGHT_V, SKY_BRIGHT_MAX_S = 150, 80    # white / grey / hazy sky
SKY_BLUE_H, SKY_BLUE_MIN_S, SKY_BLUE_MIN_V = (90, 130), 40, 80  # blue sky

THRESHOLDS = {"min_luma": 40.0, "min_sharpness": 10.0, "min_sky": 0.08}  # calibrated, see module docstring


def features(bgr: np.ndarray) -> dict:
    """One BGR uint8 frame -> {"luma", "sharpness", "sky"}."""
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (WIDTH, max(1, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    b, g, r = (small[..., i].astype(np.float32) for i in range(3))
    luma = float((0.299 * r + 0.587 * g + 0.114 * b).mean())
    lap = cv2.Laplacian(grey, cv2.CV_64F)
    th, tw = max(1, lap.shape[0] // TILES), max(1, lap.shape[1] // TILES)
    tiles = [lap[i * th:(i + 1) * th, j * tw:(j + 1) * tw].var() for i in range(TILES) for j in range(TILES)]
    sharpness = float(np.percentile(tiles, TILE_PERCENTILE))
    top = cv2.cvtColor(small[: max(1, round(small.shape[0] * SKY_TOP))], cv2.COLOR_BGR2HSV)
    hue, sat, val = top[..., 0], top[..., 1], top[..., 2]
    bright = (val >= SKY_BRIGHT_V) & (sat <= SKY_BRIGHT_MAX_S)
    blue = (hue >= SKY_BLUE_H[0]) & (hue <= SKY_BLUE_H[1]) & (sat >= SKY_BLUE_MIN_S) & (val >= SKY_BLUE_MIN_V)
    return {"luma": round(luma, 1), "sharpness": round(sharpness, 1), "sky": round(float((bright | blue).mean()), 3)}


def reasons(f: dict, t: dict = THRESHOLDS) -> list[str]:
    """Reject reasons for one frame's features (empty = pass)."""
    out = []
    if f["luma"] < t["min_luma"]:
        out.append("dark")
    if f["sharpness"] < t["min_sharpness"]:
        out.append("blur")
    if f["sky"] < t["min_sky"]:
        out.append("no_sky")
    return out


def gate(bgr_frames: list[np.ndarray], t: dict = THRESHOLDS) -> tuple[list[np.ndarray], list[str], list[dict]]:
    """-> (passing frames, sorted unique reject reasons, per-frame features + reasons for logging)."""
    kept, why, log = [], set(), []
    for frame in bgr_frames:
        f = features(frame)
        r = reasons(f, t)
        log.append({**f, "reject": r})
        if r:
            why.update(r)
        else:
            kept.append(frame)
    return kept, sorted(why, key=REASONS.index), log
