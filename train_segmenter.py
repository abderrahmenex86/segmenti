if __name__ == "__main__":
    import json
    import random

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms.v2 import (
        ColorJitter,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        RandomVerticalFlip,
        Resize,
        ToDtype,
        ToImage,
    )

    from BCEDiceLoss import BCEDiceLoss
    from datasets import PlantSegDataset
    from helpers import train
    from models import DiseaseSegmenter

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.benchmark = True

    n_epochs = 40
    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_workers = 16
    val_workers = 12
    pin_memory = True
    prefetch_factor = 4

    criterion = BCEDiceLoss(pos_weight=torch.tensor([10.0])).to(device)

    root_dir = "dataset/plantsegv3"
    n_classes = 2

    train_transforms = Compose(
        [
            ToImage(),
            Resize((520, 520)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            ColorJitter(brightness=0.3, contrast=0.3),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = PlantSegDataset(root_dir, "train", train_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    val_transforms = Compose(
        [
            ToImage(),
            Resize((512, 512)),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_dataset = PlantSegDataset(root_dir, "val", val_transforms)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    model = DiseaseSegmenter().to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.model.backbone.parameters(), "lr": 1e-5},
            {"params": model.model.classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, n_epochs, n_classes)

    with open("training_history.json", "w") as f:
        json.dump(history, f)
