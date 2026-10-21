"""PyTorch Dataset + transforms over the load_traqid() table."""

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Stored in every checkpoint so the backend can reproduce it exactly:
# RGB PIL image -> resize short side to 256 (bilinear, antialiased) ->
# centre crop 224 -> float in [0, 1] -> normalise with mean/std -> CHW.
EVAL_PREPROCESS = {
    "color": "RGB",
    "resize_short_side": 256,
    "interpolation": "bilinear",
    "antialias": True,
    "center_crop": 224,
    "scale": "divide by 255",
    "mean": list(IMAGENET_MEAN),
    "std": list(IMAGENET_STD),
    "layout": "NCHW",
}


# Smoke/fire: squash the WHOLE frame to 224x224 (no centre crop), so smoke at the frame edges
# stays visible. Backend: RGB -> resize to 224x224 (bilinear, antialiased) -> /255 -> mean/std -> NCHW.
SQUASH_PREPROCESS = {
    "color": "RGB",
    "resize_hw": [224, 224],
    "interpolation": "bilinear",
    "antialias": True,
    "scale": "divide by 255",
    "mean": list(IMAGENET_MEAN),
    "std": list(IMAGENET_STD),
    "layout": "NCHW",
}


def eval_transform(pre: dict = EVAL_PREPROCESS) -> v2.Compose:
    if "resize_hw" in pre:  # squash whole frame
        resize = [v2.Resize(pre["resize_hw"], interpolation=v2.InterpolationMode.BILINEAR, antialias=pre["antialias"])]
    else:  # short side then centre crop
        resize = [v2.Resize(pre["resize_short_side"], interpolation=v2.InterpolationMode.BILINEAR, antialias=pre["antialias"]),
                  v2.CenterCrop(pre["center_crop"])]
    return v2.Compose([
        v2.ToImage(),
        *resize,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(pre["mean"], pre["std"]),
    ])


def train_transform(size: int = 224) -> v2.Compose:
    """Whole-scene crops + strong lighting jitter (time-of-day confound, CLAUDE.md §4).

    Brightness/hue are jittered hard; contrast/saturation only moderately
    because they are haze cues themselves.
    """
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.05)], p=0.8),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.2),
        v2.RandomAdjustSharpness(2.0, p=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def smokefire_train_transform(size: int = 224) -> v2.Compose:
    """Normal augmentation for smoke/fire: flip, RandomResizedCrop 0.6-1 (aspect up to 16:9, then
    squashed to square like eval), moderate colour jitter."""
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(3 / 4, 16 / 9), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)], p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MultiLabelImages(Dataset):
    """paths + float targets (N x L) -> (image tensor, target). JPEGs decode at reduced scale (>= draft px)."""

    def __init__(self, paths: list[str], targets, transform, draft: int = 320):
        self.paths = list(paths)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.transform = transform
        self.draft = draft

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            if self.draft and im.format == "JPEG":
                im.draft("RGB", (self.draft, self.draft))
            image = im.convert("RGB")
        return self.transform(image), self.targets[i]


class TraqidImages(Dataset):
    """Rows of a load_traqid() frame -> (image tensor, band)."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.bands = df["band"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        # TRAQID files are JPEGs named .png; PIL detects the format from content.
        with Image.open(self.paths[i]) as im:
            image = im.convert("RGB")
        return self.transform(image), self.bands[i]
