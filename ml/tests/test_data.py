import pandas as pd
import pytest
import torch
from PIL import Image

from hazeml.data import EVAL_PREPROCESS, TraqidImages, eval_transform, train_transform
from hazeml.evaluate import report
from hazeml.model import build_model


@pytest.fixture
def frames(tmp_path) -> pd.DataFrame:
    # Synthetic 640x360 JPEGs saved with a .png name, like TRAQID.
    rows = []
    for i, colour in enumerate([(200, 200, 210), (120, 110, 100)]):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (640, 360), colour).save(path, format="JPEG")
        rows.append({"path": str(path), "band": i})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("transform", [train_transform(), eval_transform()])
def test_dataset_shapes(frames, transform):
    x, y = TraqidImages(frames, transform)[1]
    assert x.shape == (3, 224, 224)
    assert x.dtype == torch.float32
    assert y == 1


def test_eval_transform_is_deterministic(frames):
    ds = TraqidImages(frames, eval_transform())
    assert torch.equal(ds[0][0], ds[0][0])


def test_eval_preprocess_is_serialisable():
    assert EVAL_PREPROCESS["resize_short_side"] == 256
    assert EVAL_PREPROCESS["center_crop"] == 224
    assert len(EVAL_PREPROCESS["mean"]) == len(EVAL_PREPROCESS["std"]) == 3


@pytest.mark.parametrize("arch", ["efficientnet_b0", "mobilenet_v3_large"])
def test_model_has_three_outputs(arch):
    model = build_model(arch, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 3)


def test_report_groups():
    df = pd.DataFrame({"band": [0, 2, 1, 2], "hour": [9, 9, 16, 17], "date": ["d1", "d1", "d2", "d2"]})
    probs = torch.eye(3)[[0, 2, 0, 1]].numpy()
    rep = report(df, probs, majority=0)
    assert rep["overall"]["accuracy"] == pytest.approx(0.5)
    assert rep["overall"]["majority_baseline"] == pytest.approx(0.25)
    assert rep["by_time_of_day"]["morning (<11)"]["accuracy"] == 1.0
    assert rep["by_date"]["d2"]["accuracy"] == 0.0


def test_band2_recall_undefined_without_band2():
    df = pd.DataFrame({"band": [0, 1], "hour": [16, 16], "date": ["d1", "d1"]})
    rep = report(df, torch.eye(3)[[0, 1]].numpy(), majority=0)
    assert rep["overall"]["band2_recall"] is None
