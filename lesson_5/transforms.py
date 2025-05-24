# transforms.py

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def preprocess_image(image, **kwargs):
    return image / 255.0

def preprocess_mask(mask, n_classes=22, **kwargs):
    mask = np.array(mask).astype(np.int64)
    mask[mask == 255] = n_classes
    return mask

def apply_transforms(image, mask, transforms):
    image_np = np.array(image)
    mask_np = np.array(mask)
    transformed = transforms(image=image_np, mask=mask_np)
    return transformed["image"], transformed["mask"]