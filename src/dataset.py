import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


class PlantSegDataset(Dataset):
    def __init__(self, root_dir, split, transform, num_classes=116, limit_dataset=None):
        assert split in ["train", "val", "test"], "Split must be 'train', 'val', or 'test'"
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_classes = num_classes

        self.images_dir = os.path.join(root_dir, "images", split)
        self.masks_dir = os.path.join(root_dir, "annotations", split)

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Missing images folder: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Missing annotations folder: {self.masks_dir}")

        self.filenames = sorted(
            [f for f in os.listdir(self.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

        if limit_dataset is not None:
            self.filenames = self.filenames[:limit_dataset]

        self.error_log = []

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        try:
            image_path = os.path.join(self.images_dir, self.filenames[idx])
            image = Image.open(image_path).convert("RGB")
            image = tv_tensors.Image(image)

            base_name = os.path.splitext(self.filenames[idx])[0]
            mask_path = os.path.join(self.masks_dir, f"{base_name}.png")

            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)

            if self.num_classes == 1:
                mask_np = (mask_np > 0).astype(np.float32)
            else:
                mask_np = mask_np.astype(np.int64)

            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
            mask = tv_tensors.Mask(mask_tensor)

            if self.transform:
                image, mask = self.transform(image, mask)

        except Exception as e:
            self.error_log.append(
                {"index": idx, "error": str(e), "path": image_path if "image_path" in locals() else "N/A"}
            )
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        return image, mask

    def get_error_log(self):
        return self.error_log


def get_transforms(img_size: int, num_classes: int, is_train: bool):
    mask_dtype = torch.float32 if num_classes == 1 else torch.int64

    if is_train:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((img_size, img_size), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(brightness=0.3, contrast=0.3),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        tv_tensors.Mask: mask_dtype,
                    },
                    scale=True,
                ),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((img_size, img_size), antialias=True),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        tv_tensors.Mask: mask_dtype,
                    },
                    scale=True,
                ),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def get_dataloaders(
    root_dir: str,
    batch_size: int,
    num_classes: int,
    img_size: int,
    limit_dataset: int = None,
):
    prefetch_factor = 2

    train_transform = get_transforms(img_size, num_classes, is_train=True)
    val_transform = get_transforms(img_size, num_classes, is_train=False)

    train_dataset = PlantSegDataset(root_dir, "train", train_transform, num_classes, limit_dataset)
    val_dataset = PlantSegDataset(root_dir, "val", val_transform, num_classes, limit_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    return train_loader, val_loader
