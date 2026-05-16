import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors


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

    def __getitem__(self, idx):
        try:
            image_path = os.path.join(self.images_dir, self.filenames[idx])
            image = Image.open(image_path).convert("RGB")
            image = tv_tensors.Image(image)

            mask_path = os.path.join(self.masks_dir, self.filenames[idx].replace(".jpg", ".png"))
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask).astype(np.int64)
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
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        return image, mask

    def _get_error_log(self):
        if not self.error_log:
            print("No errors encountered during loading.")
        else:
            print(f"Total errors: {len(self.error_log)}")
            for error in self.error_log:
                print(f"Index: {error['index']}, Error: {error['error']}, Path: {error['path']}")
