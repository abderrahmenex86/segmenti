import json

import torch
from tqdm.auto import tqdm


def compute_iou(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum((1, 2, 3))
    union = (preds + targets).sum((1, 2, 3)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)

    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        train_loss += loss.item() * bs
        total_samples += bs

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return train_loss / total_samples


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    total_samples = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} Val", leave=False)

    with torch.no_grad():
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            masks = masks.to(device, dtype=torch.float32)
            outputs = model(images)

            loss = criterion(outputs, masks)
            iou = compute_iou(outputs, masks)

            bs = images.size(0)
            val_loss += loss.item() * bs
            val_iou += iou * bs
            total_samples += bs

            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}"}
            )

    return val_loss / total_samples, val_iou / total_samples


def train(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs
):
    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_iou = 0.0

    print(f"Starting training on {device}...")
    progress_bar = tqdm(range(1, num_epochs + 1), unit="epoch")

    for epoch in progress_bar:
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_iou = evaluate(
            model, val_loader, criterion, device, epoch
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        tqdm.write(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_disease_segmenter.pth")
            tqdm.write(f"--> Saved new best model (IoU: {best_iou:.4f})")

        with open("segmentation_history.json", "w") as f:
            json.dump(history, f)

    return history


def parse_disease_name(filename):
    import re

    name = filename.strip("'\" \n").replace(".jpg", "")

    name = re.sub(r"_[Bb]ing_.*$", "", name)
    name = re.sub(r"_[Gg]oogle_.*$", "", name)
    name = re.sub(r"_[Bb]aidu_.*$", "", name)

    name = re.sub(r"_banana black sigatoka \(\d+\)$", "", name)
    name = re.sub(r"_blotch \(\d+\)$", "", name)
    name = re.sub(r"_black_chaff \(\d+\)$", "", name)

    name = re.sub(r"_\d+$", "", name)

    return name


def build_taxonomy(images_dir, output_file="class_mapping.json"):
    import json
    import os

    unique_classes = set()

    for filename in os.listdir(images_dir):
        if not filename.endswith(".jpg"):
            continue

        clean_name = parse_disease_name(filename)
        unique_classes.add(clean_name)

    sorted_classes = sorted(list(unique_classes))

    class_to_id = {name: idx for idx, name in enumerate(sorted_classes)}

    with open(output_file, "w") as f:
        json.dump(class_to_id, f, indent=4)

    print(f"Saved to {output_file}")
