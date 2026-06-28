from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


class SegmentDataset(Dataset):
    def __init__(self, root, split, transform, num_classes, limit=None):
        self.img_dir, self.mask_dir = Path(root) / "images" / split, Path(root) / "annotations" / split
        self.files = sorted([f.name for f in self.img_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".png"]])[:limit]
        self.transform, self.num_classes = transform, num_classes

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.img_dir / self.files[idx]
        mask_numpy = np.array(Image.open(self.mask_dir / f"{image_path.stem}.png").convert("L"))
        mask_tensor = torch.from_numpy(
            (mask_numpy > 0).astype(np.float32) if self.num_classes == 1 else mask_numpy.astype(np.int64)
        ).unsqueeze(0)

        return self.transform(tv_tensors.Image(Image.open(image_path).convert("RGB")), tv_tensors.Mask(mask_tensor))


def get_dataloaders(root_dir, batch_size, img_size, num_classes, limit_dataset=None, **kwargs):
    mask_dtype = torch.float32 if num_classes == 1 else torch.int64

    base_transforms = [v2.ToImage(), v2.Resize((img_size, img_size), antialias=True)]
    normalization_transforms = [
        v2.ToDtype(dtype={tv_tensors.Image: torch.float32, tv_tensors.Mask: mask_dtype}, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    train_transforms = v2.Compose(
        base_transforms
        + [
            v2.RandomHorizontalFlip(p=kwargs.get("aug_hflip_p", 0.5)),
            v2.RandomVerticalFlip(p=kwargs.get("aug_vflip_p", 0.5)),
            v2.RandomRotation(degrees=kwargs.get("aug_rotation", 25)),
            v2.ColorJitter(
                brightness=kwargs.get("aug_brightness", 0.2),
                contrast=kwargs.get("aug_contrast", 0.2),
                saturation=kwargs.get("aug_saturation", 0.2),
            ),
        ]
        + normalization_transforms
    )

    val_transforms = v2.Compose(base_transforms + normalization_transforms)

    use_cuda = torch.cuda.is_available()

    num_workers_train = kwargs.get("train_num_workers", 10 if use_cuda else 0)
    num_workers_val = kwargs.get("val_num_workers", 4 if use_cuda else 0)
    prefetch_factor_value = kwargs.get("prefetch_factor", 4)
    use_pin_memory = kwargs.get("pin_memory", use_cuda)
    use_persistent_workers = kwargs.get("persistent_workers", use_cuda)

    train_loader = DataLoader(
        SegmentDataset(root_dir, "train", train_transforms, num_classes, limit_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=use_pin_memory,
        prefetch_factor=prefetch_factor_value if num_workers_train > 0 else None,
        persistent_workers=use_persistent_workers if num_workers_train > 0 else False,
    )

    val_loader = DataLoader(
        SegmentDataset(root_dir, "val", val_transforms, num_classes, limit_dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_val,
        pin_memory=use_pin_memory,
        prefetch_factor=prefetch_factor_value if num_workers_val > 0 else None,
        persistent_workers=use_persistent_workers if num_workers_val > 0 else False,
    )

    return train_loader, val_loader
