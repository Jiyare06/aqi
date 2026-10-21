"""Export pretrained YOLO11n (COCO) to ONNX and check our numpy postprocess against ultralytics.

    uv run python -m hazeml.yolo_export

Writes ml/runs/yolo/ (gitignored): yolo11n.pt (downloaded by ultralytics), yolo11n.onnx,
export_report.json. Frames: 20 TRAQID front frames picked from the CSV (one per date + 3 more, seed 0).
Ultralytics YOLO11n is AGPL-3.0.
"""

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install from inside the run

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402

from hazeml import vehicles  # noqa: E402
from hazeml.traqid import load_traqid  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "runs" / "yolo"


def pick_frames(n_extra: int = 3, seed: int = 0):
    df = load_traqid()  # day, front; reads only the CSV
    one_per_date = df.groupby("date", group_keys=False).apply(lambda g: g.sample(1, random_state=seed))
    extra = df.drop(one_per_date.index).sample(n_extra, random_state=seed)
    return list(one_per_date["path"]) + list(extra["path"])


def ul_counts(result) -> dict[str, int]:
    counts = {name: 0 for name in vehicles.VEHICLE_CLASSES.values()}
    for c in result.boxes.cls.tolist():
        counts[vehicles.VEHICLE_CLASSES[int(c)]] += 1
    return counts


def match_boxes(ours: list[dict], ul) -> dict:
    """Pair our boxes with ultralytics' (same class, best IoU) -> max coordinate / confidence differences."""
    theirs = [(int(c), float(s), b) for c, s, b in zip(ul.boxes.cls.tolist(), ul.boxes.conf.tolist(), ul.boxes.xyxy.tolist())]
    dxy, dconf, unmatched = 0.0, 0.0, 0
    for d in ours:
        cands = [(abs(np.array(b) - np.array(d["xyxy"])).max(), abs(s - d["conf"])) for c, s, b in theirs if c == d["class_id"]]
        if not cands:
            unmatched += 1
            continue
        best = min(cands)
        dxy, dconf = max(dxy, best[0]), max(dconf, best[1])
    return {"max_box_px_diff": dxy, "max_conf_diff": dconf, "unmatched": unmatched}


def main() -> None:
    from ultralytics import YOLO, settings
    settings.update({"sync": False})  # no usage analytics

    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT)  # ultralytics downloads yolo11n.pt into the working directory
    pt = YOLO("yolo11n.pt")
    onnx_path = Path(pt.export(format="onnx", imgsz=vehicles.IMGSZ, opset=17, simplify=False, dynamic=False, half=False))
    ul_onnx = YOLO(str(onnx_path), task="detect")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
    run = lambda x: sess.run(None, {inp.name: x})[0]  # noqa: E731

    kw = dict(imgsz=vehicles.IMGSZ, conf=vehicles.CONF, iou=vehicles.IOU, classes=list(vehicles.VEHICLE_CLASSES),
              device="cpu", verbose=False)
    frames, per_frame = pick_frames(), []
    for path in frames:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        ours = vehicles.detect(run, rgb)
        r_onnx = ul_onnx.predict(path, **kw)[0]
        r_pt = pt.predict(path, **kw)[0]
        c_ours, c_onnx, c_pt = vehicles.count_by_class(ours), ul_counts(r_onnx), ul_counts(r_pt)
        per_frame.append({"file": Path(path).name, "size": list(rgb.shape[:2]), "ours": c_ours,
                          "ultralytics_onnx": c_onnx, "ultralytics_pt": c_pt,
                          "total": {"ours": sum(c_ours.values()), "ultralytics_onnx": sum(c_onnx.values()),
                                    "ultralytics_pt": sum(c_pt.values())},
                          "box_match_vs_ultralytics_onnx": match_boxes(ours, r_onnx)})

    # CPU timing on one frame (preprocess / ONNX inference / postprocess), default threads and 1 thread.
    rgb = np.asarray(Image.open(frames[0]).convert("RGB"))
    timing = {}
    for name, threads in (("default_threads", 0), ("one_thread", 1)):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        s = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        pre, inf, post = [], [], []
        for k in range(25):
            t0 = time.perf_counter()
            x, r, pad = vehicles.letterbox(rgb)
            t1 = time.perf_counter()
            y = s.run(None, {inp.name: x})[0]
            t2 = time.perf_counter()
            b, sc, c = vehicles.decode(y)
            vehicles.class_aware_nms(b, sc, c)
            t3 = time.perf_counter()
            if k >= 5:  # skip warm-up
                pre.append(t1 - t0), inf.append(t2 - t1), post.append(t3 - t2)
        timing[name] = {"preprocess_ms": 1000 * float(np.median(pre)), "inference_ms": 1000 * float(np.median(inf)),
                        "postprocess_ms": 1000 * float(np.median(post)),
                        "total_ms": 1000 * float(np.median(np.array(pre) + np.array(inf) + np.array(post)))}
    timing["cpu"] = {"os_cpu_count": os.cpu_count(), "frame_size_hw": list(rgb.shape[:2]), "runs": 20}

    n = len(per_frame)
    tot = lambda k: [f["total"][k] for f in per_frame]  # noqa: E731
    report = {
        "model": "yolo11n.pt (Ultralytics, COCO-pretrained, AGPL-3.0)", "onnx": onnx_path.name,
        "onnx_mb": onnx_path.stat().st_size / 2**20, "opset": 17, "onnxruntime": ort.__version__,
        "input": {"name": inp.name, "shape": inp.shape, "type": inp.type, "layout": "NCHW RGB float32 in [0,1], letterboxed, pad 114"},
        "output": {"name": outp.name, "shape": outp.shape,
                   "layout": "1 x (4 + 80) x anchors: rows 0-3 = cx, cy, w, h in letterboxed pixels; rows 4-83 = class scores (sigmoid, no objectness)"},
        "thresholds": {"conf": vehicles.CONF, "iou": vehicles.IOU, "classes": vehicles.VEHICLE_CLASSES, "max_det": vehicles.MAX_DET},
        "clip_rule": "vehicle_count = median of per-frame counts over the sampled frames (rounded half up)",
        "parity": {
            "frames": n,
            "per_class_counts_equal_vs_ultralytics_onnx": sum(f["ours"] == f["ultralytics_onnx"] for f in per_frame),
            "per_class_counts_equal_vs_ultralytics_pt": sum(f["ours"] == f["ultralytics_pt"] for f in per_frame),
            "total_counts": {"ours": tot("ours"), "ultralytics_onnx": tot("ultralytics_onnx"), "ultralytics_pt": tot("ultralytics_pt")},
            "max_box_px_diff_vs_ultralytics_onnx": max(f["box_match_vs_ultralytics_onnx"]["max_box_px_diff"] for f in per_frame),
            "max_conf_diff_vs_ultralytics_onnx": max(f["box_match_vs_ultralytics_onnx"]["max_conf_diff"] for f in per_frame),
            "clip_vehicle_count_over_these_frames": {"ours": vehicles.clip_vehicle_count(tot("ours"))},
            "note": "ultralytics_pt letterboxes to the minimal stride-32 rectangle (640x384 for 640x360 frames); the ONNX path uses 640x640",
        },
        "per_frame": per_frame,
        "timing_cpu": timing,
        "licence": "Ultralytics YOLO11n weights and code are AGPL-3.0; using them requires this repo to stay open source under AGPL-compatible terms.",
    }
    (OUT / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
