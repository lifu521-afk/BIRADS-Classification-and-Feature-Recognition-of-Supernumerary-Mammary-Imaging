import os
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_TO_ID = {
    "2类": 0,
    "3类": 1,
    "4A类": 2,
    "4B类": 3,
    "4C类": 4,
    "5类": 5,
}


class ClassificationDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            images_path = os.path.join(class_path, "images")
            if not os.path.isdir(images_path) or class_name not in CLASS_TO_ID:
                continue
            class_id = CLASS_TO_ID[class_name]
            for image_name in os.listdir(images_path):
                full_path = os.path.join(images_path, image_name)
                if os.path.isfile(full_path):
                    self.samples.append((full_path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class FeatureDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, "images")
        self.label_dirs = [
            os.path.join(root_dir, "boundary_labels"),
            os.path.join(root_dir, "calcification_labels"),
            os.path.join(root_dir, "direction_labels"),
            os.path.join(root_dir, "shape_labels"),
        ]
        self.samples: List[Tuple[str, torch.Tensor]] = []
        for image_name in sorted(os.listdir(self.images_dir)):
            image_path = os.path.join(self.images_dir, image_name)
            image_stem, _ = os.path.splitext(image_name)
            label_paths = [os.path.join(label_dir, f"{image_stem}.txt") for label_dir in self.label_dirs]
            if not all(os.path.exists(path) and os.path.getsize(path) > 0 for path in label_paths):
                continue
            labels = [self._read_binary_label(path) for path in label_paths]
            self.samples.append((image_path, torch.tensor(labels, dtype=torch.float32)))

    def _read_binary_label(self, label_path: str) -> int:
        frame = pd.read_csv(label_path, sep=r"\s+", header=None)
        values = frame.iloc[:, 0]
        return int(values.sum() >= 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, labels = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, labels
