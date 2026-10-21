import io

import numpy as np
import pytest
from PIL import Image

from hazeml import dfire


def jpeg(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_dhash_near_duplicate_close_and_different_image_far():
    rng = np.random.default_rng(0)
    base = np.tile(np.linspace(0, 255, 320), (240, 1))[..., None].repeat(3, axis=2)
    noisy = np.clip(base + rng.normal(0, 3, base.shape), 0, 255)
    other = rng.integers(0, 256, base.shape)
    h0, h1, h2 = dfire.dhash(jpeg(base)), dfire.dhash(jpeg(noisy)), dfire.dhash(jpeg(other))
    assert bin(h0 ^ h1).count("1") <= 4
    assert bin(h0 ^ h2).count("1") > 16


def test_groups_are_connected_components():
    h = np.array([0b0000, 0b0001, 0b0011, 0xFFFF0000, 0xFFFF0001], dtype=np.uint64)
    g, _ = dfire.groups(h, max_dist=1)
    assert g[0] == g[1] == g[2]  # 0-1 and 1-2 are 1 bit apart -> chained
    assert g[3] == g[4] and g[3] != g[0]
    assert len(set(dfire.groups(h, max_dist=0)[0].tolist())) == 5


def test_extra_links_respect_cap():
    h = np.array([1, 2, 4, 8, 16, 32], dtype=np.uint64)  # all >= 2 bits apart
    g, stats = dfire.groups(h, max_dist=0, extra_pairs=[(0, 1), (1, 2), (3, 4)], max_share=0.34)  # cap = 2 images
    assert g[0] == g[1] and g[2] != g[0] and g[3] == g[4]
    assert stats["extra_merged"] == 2 and stats["extra_skipped_by_cap"] == 1


def test_aof_pairs_only_consecutive():
    assert dfire.aof_pairs(["AoF00001.jpg", "WEB00002.jpg", "AoF00002.jpg", "AoF00004.jpg"]) == [(0, 2)]


def test_class_ids_inferred_from_published_box_counts():
    rows = [{"label_text": "0 .5 .5 .1 .1\n" * 14692}, {"label_text": "1 .5 .5 .1 .1\n" * 11865}]
    assert dfire.class_ids(rows) == {"fire": 0, "smoke": 1}
    with pytest.raises(SystemExit):
        dfire.class_ids([{"label_text": "0 .5 .5 .1 .1\n"}])


def test_labels_from_boxes():
    rows = [{"label_text": ""}, {"label_text": "1 .5 .5 .1 .1"}, {"label_text": "0 .1 .1 .1 .1"},
            {"label_text": "0 .1 .1 .1 .1\n1 .2 .2 .1 .1"}]
    dfire.add_labels(rows, {"fire": 0, "smoke": 1})
    assert [r["label"] for r in rows] == ["clear", "smoke", "fire", "smoke_fire"]
