"""Class activation maps (CAM) in numpy, from the ONNX model's outputs. No torch needed (Lambda-friendly).

The smoke/fire model is EfficientNet-B0 -> global average pool -> Linear(1280 -> 2). So for label c,
    logit_c = mean_{h,w}( sum_k W[c,k] * F[k,h,w] ) + b[c],
and the CAM is the spatial map inside that mean: CAM_c[h,w] = sum_k W[c,k] * F[k,h,w]
(Zhou et al. 2016, "Learning Deep Features for Discriminative Localization").
`features` is the final conv feature map (1280 x 7 x 7 for a 224 input); W, b come from head.npz.
"""

import numpy as np


def cam_raw(features: np.ndarray, weight: np.ndarray, label: int) -> np.ndarray:
    """(C,H,W) features, (L,C) weight -> (H,W) un-normalised CAM for `label`."""
    return np.tensordot(weight[label], features, axes=([0], [0]))


def cam(features: np.ndarray, weight: np.ndarray, label: int) -> np.ndarray:
    """CAM clipped at 0 and scaled to [0, 1] (all zeros if nothing is positive)."""
    m = np.maximum(cam_raw(features, weight, label), 0)
    return m / m.max() if m.max() > 0 else m


def resize_bilinear(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize of a 2-D array (align corners off, like image resizing)."""
    def coords(n_out, n_in):
        x = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
        return np.clip(x, 0, n_in - 1)
    ys, xs = coords(h, a.shape[0]), coords(w, a.shape[1])
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, a.shape[0] - 1), np.minimum(x0 + 1, a.shape[1] - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bottom = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a [0,1] heat map (any size) over an HxWx3 uint8 image as a red-yellow colour ramp."""
    h = resize_bilinear(heat, rgb.shape[0], rgb.shape[1])[..., None]
    colour = np.concatenate([np.ones_like(h), h, np.zeros_like(h)], axis=2) * 255  # red -> yellow as heat rises
    a = alpha * h
    return (rgb * (1 - a) + colour * a).clip(0, 255).astype(np.uint8)
