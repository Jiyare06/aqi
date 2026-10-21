import numpy as np
import pytest

from hazeml import vehicles


def test_letterbox_640x360_pads_top_bottom():
    x, r, (left, top) = vehicles.letterbox(np.zeros((360, 640, 3), dtype=np.uint8))
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert (r, left, top) == (1.0, 0, 140)
    assert x[0, :, 0, 0] == pytest.approx(114 / 255) and x[0, :, 320, 320] == pytest.approx(0)


def test_letterbox_resizes_large_frames():
    x, r, (left, top) = vehicles.letterbox(np.full((1080, 1920, 3), 200, dtype=np.uint8))
    assert r == pytest.approx(1 / 3) and (left, top) == (0, 140) and x[0, 0, 320, 320] == pytest.approx(200 / 255)


def fake_output(rows):
    """rows: (cx, cy, w, h, class_id, score) -> 1x84xN."""
    out = np.zeros((1, 84, len(rows)), dtype=np.float32)
    for j, (cx, cy, w, h, c, s) in enumerate(rows):
        out[0, :4, j] = cx, cy, w, h
        out[0, 4 + c, j] = s
    return out


def test_decode_uses_best_class_then_filters_vehicles():
    out = fake_output([(100, 100, 20, 20, 2, 0.9), (200, 200, 20, 20, 0, 0.9), (300, 300, 20, 20, 7, 0.35)])
    out[0, 4 + 2, 1] = 0.8  # a person box with car as second-best class is dropped (best class wins)
    boxes, scores, cls = vehicles.decode(out)
    assert cls.tolist() == [2] and boxes[0].tolist() == [90, 90, 110, 110]  # score 0.35 is not > 0.35


def test_nms_is_per_class():
    out = fake_output([(100, 100, 40, 40, 2, 0.9), (102, 100, 40, 40, 2, 0.8), (102, 100, 40, 40, 7, 0.7)])
    b, s, c = vehicles.decode(out)
    keep = vehicles.class_aware_nms(b, s, c)
    assert sorted(c[keep].tolist()) == [2, 7]


def test_detect_maps_back_to_original_pixels():
    out = fake_output([(320, 320, 64, 40, 3, 0.9)])
    dets = vehicles.detect(lambda x: out, np.zeros((360, 640, 3), dtype=np.uint8))
    assert dets[0]["cls"] == "motorcycle" and dets[0]["xyxy"] == pytest.approx([288, 160, 352, 200])
    assert vehicles.count_by_class(dets) == {"car": 0, "motorcycle": 1, "bus": 0, "truck": 0}


@pytest.mark.parametrize("counts, expected", [([], 0), ([3], 3), ([2, 3], 3), ([1, 4, 9], 4), ([0, 0, 5, 7], 3)])
def test_clip_rule_is_median_rounded_half_up(counts, expected):
    assert vehicles.clip_vehicle_count(counts) == expected
