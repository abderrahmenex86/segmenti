import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm.auto import tqdm

from datasets import PlantDiseaseDataset
from models import DiseaseClassifier


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)

    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / total, (correct / total) * 100


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch} Val", leave=False)

    with torch.no_grad():
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return running_loss / total, (correct / total) * 100


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 20
    batch_size = 32

    train_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(224, antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(15),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    root_dir = "dataset/plantsegv3"
    train_dataset = PlantDiseaseDataset(root_dir, "train", train_transforms)
    val_dataset = PlantDiseaseDataset(root_dir, "val", val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model = DiseaseClassifier(num_classes=115).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.classifier_head.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, epoch
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_disease_classifier.pth")
            print(f"--> Saved new best model (Acc: {best_acc:.1f}%)")

        with open("classification_history.json", "w") as f:
            json.dump(history, f)
