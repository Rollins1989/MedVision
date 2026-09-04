"""
src/preprocessing/transforms.py
--------------------------------
Preprocessing (resize, normalize) and training-time augmentation, built on
torchvision.transforms.v2. Kept deliberately simple and inspectable rather
than pulling in a second augmentation library.
"""
from dataclasses import dataclass

import torch
import torchvision.transforms.v2 as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True
    rotation_degrees: int = 10
    brightness_contrast: bool = True


def build_transforms(image_size: int, train: bool, aug: AugmentationConfig | None = None) -> T.Compose:
    """Grayscale chest X-rays are replicated to 3 channels so ImageNet-
    pretrained backbones (which expect 3-channel input) can be used
    directly via transfer learning."""
    aug = aug or AugmentationConfig()

    ops = [
        T.Grayscale(num_output_channels=3),
        T.Resize((image_size, image_size)),
    ]

    if train:
        if aug.horizontal_flip:
            ops.append(T.RandomHorizontalFlip(p=0.5))
        if aug.rotation_degrees > 0:
            ops.append(T.RandomRotation(degrees=aug.rotation_degrees))
        if aug.brightness_contrast:
            ops.append(T.ColorJitter(brightness=0.15, contrast=0.15))

    ops += [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(ops)
