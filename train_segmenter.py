import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from BCEDiceLoss import BCEDiceLoss
from datasets import PlantSegDataset
from helpers import train
from models import DiseaseSegmenter

if __name__ == "__main__":
    num_epochs = 20
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((256, 256), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((256, 256), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    root_dir = "dataset/plantsegv3"

    train_dataset = PlantSegDataset(root_dir, "train", train_transforms)
    val_dataset = PlantSegDataset(root_dir, "val", val_transforms)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    model = DiseaseSegmenter().to(device)
    criterion = BCEDiceLoss(pos_weight=torch.tensor([2.0]).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs,
    )
