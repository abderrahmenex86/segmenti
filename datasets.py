import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from helpers import parse_disease_name


class PlantSegDataset(Dataset):
    def __init__(self, root_dir, split, transform):
        assert split in [
            "train",
            "val",
            "test",
        ], "Split must be 'train', 'val', or 'test'"
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, "images", split)
        self.masks_dir = os.path.join(root_dir, "annotations", split)
        self.filenames = sorted(os.listdir(self.images_dir))

        self.error_log = []

    def __len__(self):
        return len(self.filenames)

    def get_error_log(self):
        if not self.error_log:
            print("No errors encountered during loading.")
        else:
            print(f"Total errors: {len(self.error_log)}")
            for error in self.error_log:
                print(
                    f"Index: {error['index']}, Error: {error['error']}, Path: {error['path']}"
                )

    def __getitem__(self, idx):
        try:
            image_path = os.path.join(self.images_dir, self.filenames[idx])
            image = Image.open(image_path)
            if image.mode != "RGB":
                raise ValueError(f"Image {image_path} is not RGB: {image.mode}")

            mask_path = os.path.join(
                self.masks_dir, self.filenames[idx].replace(".jpg", ".png")
            )

            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
            mask = (mask > 0).astype(np.uint8)
            mask = torch.from_numpy(mask).unsqueeze(0)
            mask = tv_tensors.Mask(mask)

            if self.transform:
                image, mask = self.transform(image, mask)

        except Exception as e:
            self.error_log.append(
                {
                    "index": idx,
                    "error": str(e),
                    "path": image_path if "image_path" in locals() else "N/A",
                }
            )
            print(f"Error loading image at index {idx}: {e}")
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        return image, mask


class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, split, transform):
        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, "images", split)
        self.transform = transform

        with open("class_mapping.json", "r") as f:
            self.class_to_id = json.load(f)

        self.filenames = [
            f for f in sorted(os.listdir(self.images_dir)) if f.endswith(".jpg")
        ]

        self.error_log = []

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        try:
            filename = self.filenames[idx]
            image_path = os.path.join(self.images_dir, filename)

            image = Image.open(image_path).convert("RGB")

            name = parse_disease_name(filename)
            label_id = self.class_to_id[name]

            if self.transform:
                image = self.transform(image)
        except Exception as e:
            self.error_log.append(
                {
                    "index": idx,
                    "error": str(e),
                    "path": image_path if "image_path" in locals() else "N/A",
                }
            )
            print(f"Error loading image at index {idx}: {e}")
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        return image, torch.tensor(label_id, dtype=torch.long)
