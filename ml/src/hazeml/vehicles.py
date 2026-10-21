"""Vehicle count from pretrained YOLO11n (COCO) ONNX output — numpy only (no torch / ultralytics / cv2).

Model: Ultralytics YOLO11n, **AGPL-3.0** (https://github.com/ultralytics/ultralytics). Exported with
hazeml.yolo_export (imgsz 640, opset 17). ONNX input "images" 1x3x640x640 float32 RGB in [0,1];
output "output0" 1x84x8400 = per anchor [cx, cy, w, h] (letterboxed pixels) + 80 class scores (already sigmoid).

Pipeline (mirrors ultralytics' predict for an exported model):
1. letterbox: scale by r = min(640/h, 640/w), resize (bilinear, half-pixel), pad with 114 to 640x640, centred.
2. decode: per anchor take the best of all 80 classes; keep if score > conf; then keep only vehicle classes.
3. NMS per class (greedy, IoU > iou suppresses), max 300 detections.
Per clip: vehicle_count = median of the per-frame counts over the sampled frames (rounded half up).
"""

import numpy as np

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONF, IOU, IMGSZ, PAD = 0.35, 0.5, 640, 114
MAX_DET = 300


def resize_bilinear(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """uint8 HxWxC -> uint8 hxwxC, bilinear with half-pixel centres (like cv2.INTER_LINEAR)."""
    if img.shape[:2] == (h, w):
        return img

    def coords(n_out, n_in):
        x = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
        x0 = np.floor(x).astype(int)
        return x0, np.minimum(x0 + 1, n_in - 1), (x - x0)

    y0, y1, wy = coords(h, img.shape[0])
    x0, x1, wx = coords(w, img.shape[1])
    a = img.astype(np.float32)
    wx, wy = wx[None, :, None], wy[:, None, None]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return np.clip(np.round(top * (1 - wy) + bot * wy), 0, 255).astype(np.uint8)


def letterbox(rgb: np.ndarray, size: int = IMGSZ) -> tuple[np.ndarray, float, tuple[int, int]]:
    """uint8 HxWx3 RGB -> (1x3xSxS float32 in [0,1], scale r, (left, top) padding)."""
    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = np.full((nh + top + bottom, nw + left + right, 3), PAD, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resize_bilinear(rgb, nh, nw)
    x = out.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, (left, top)


def decode(output: np.ndarray, conf: float = CONF, classes=tuple(VEHICLE_CLASSES)) -> tuple[np.ndarray, ...]:
    """1x84xN (or 84xN) -> (xyxy boxes Kx4 in letterboxed px, scores K, class ids K)."""
    pred = output[0] if output.ndim == 3 else output
    scores_all = pred[4:]
    cls = scores_all.argmax(0)
    score = scores_all[cls, np.arange(pred.shape[1])]
    keep = (score > conf) & np.isin(cls, classes)
    cx, cy, w, h = pred[:4, keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, score[keep], cls[keep]


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = IOU) -> np.ndarray:
    """Greedy NMS; indices kept, highest score first."""
    order = np.argsort(-scores, kind="stable")
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1, yy1 = np.maximum(boxes[i, 0], boxes[rest, 0]), np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2, yy2 = np.minimum(boxes[i, 2], boxes[rest, 2]), np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        overlap = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[overlap <= iou]
    return np.array(keep, dtype=int)


def class_aware_nms(boxes, scores, cls, iou: float = IOU, max_det: int = MAX_DET) -> np.ndarray:
    keep = np.concatenate([np.flatnonzero(cls == c)[nms(boxes[cls == c], scores[cls == c], iou)] for c in np.unique(cls)]) \
        if len(cls) else np.array([], dtype=int)
    return keep[np.argsort(-scores[keep], kind="stable")][:max_det]


def detect(run_onnx, rgb: np.ndarray, conf: float = CONF, iou: float = IOU) -> list[dict]:
    """run_onnx: callable (1x3x640x640 float32) -> 1x84xN output. Boxes returned in original image pixels."""
    x, r, (left, top) = letterbox(rgb)
    boxes, scores, cls = decode(run_onnx(x), conf)
    keep = class_aware_nms(boxes, scores, cls, iou)
    h, w = rgb.shape[:2]
    out = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        xyxy = [float(np.clip((x1 - left) / r, 0, w)), float(np.clip((y1 - top) / r, 0, h)),
                float(np.clip((x2 - left) / r, 0, w)), float(np.clip((y2 - top) / r, 0, h))]
        out.append({"cls": VEHICLE_CLASSES[int(cls[i])], "class_id": int(cls[i]), "conf": float(scores[i]), "xyxy": xyxy})
    return out


def count_by_class(dets: list[dict]) -> dict[str, int]:
    counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    for d in dets:
        counts[d["cls"]] += 1
    return counts


def clip_vehicle_count(per_frame_counts: list[int]) -> int:
    """Proposed per-clip rule: median of per-frame vehicle counts, rounded half up (0 if no frames)."""
    return int(np.floor(np.median(per_frame_counts) + 0.5)) if len(per_frame_counts) else 0
