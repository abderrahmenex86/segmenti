import json

import torch
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex
from tqdm.auto import tqdm


def compute_iou(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum((1, 2, 3))
    union = (preds + targets).sum((1, 2, 3)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def train_epoch(model, loader, criterion, optimizer, device, dice, iou):

    model.train()

    dice.reset()
    iou.reset()

    total_loss = 0.0
    total_samples = 0

    for images, masks in tqdm(loader, desc="Training", leave=False, unit="batch"):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(predictions, masks)

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        probs = torch.softmax(predictions["out"], dim=1)

        dice.update(probs, masks.squeeze(1).long())
        iou.update(probs, masks.squeeze(1).long())

    epoch_dice = dice.compute().item()
    epoch_iou = iou.compute().item()

    return {
        "loss": total_loss / total_samples,
        "dice": epoch_dice * 100,
        "iou": epoch_iou * 100,
    }


def evaluate(model, loader, criterion, device, dice, iou):

    model.eval()

    dice.reset()
    iou.reset()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating", leave=False, unit="batch"):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            predictions = model(images)

            loss = criterion(predictions, masks)

            batch_size = images.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            probs = torch.softmax(predictions, dim=1)

            dice.update(probs, masks.squeeze(1).long())
            iou.update(probs, masks.squeeze(1).long())

    epoch_dice = dice.compute().item()
    epoch_iou = iou.compute().item()

    return {
        "loss": total_loss / total_samples,
        "dice": epoch_dice * 100,
        "iou": epoch_iou * 100,
    }


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, n_epochs, n_classes):
    best_val_dice = 0.0
    min_delta = 1e-4
    history = {
        "train": {"loss": [], "dice": [], "iou": []},
        "val": {"loss": [], "dice": [], "iou": []},
    }

    dice = MulticlassF1Score(num_classes=n_classes, average="macro").to(device)
    iou = MulticlassJaccardIndex(num_classes=n_classes, average="macro").to(device)

    for epoch in tqdm(range(1, n_epochs + 1), unit="epoch", leave=True):

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, dice, iou)
        val_metrics = evaluate(model, val_loader, criterion, device, dice, iou)

        scheduler.step(val_metrics["dice"])

        current_lr = optimizer.param_groups[1]["lr"]
        tqdm.write(f"Current LR: {current_lr:.6f}")

        for k, v in train_metrics.items():
            history["train"][k].append(v)
        for k, v in val_metrics.items():
            history["val"][k].append(v)

        tqdm.write(
            f"Train Set | Loss -> {train_metrics['loss']:.4f} | Dice Score -> {train_metrics['dice']:.2f}% | IoU -> {train_metrics['iou']:.2f}%"
        )

        tqdm.write(
            f"Val Set   | Loss -> {val_metrics['loss']:.4f} | Dice Score -> {val_metrics['dice']:.2f}% | IoU -> {val_metrics['iou']:.2f}%"
        )

        current_val_dice = val_metrics["dice"]

        if current_val_dice > best_val_dice + min_delta:

            best_val_dice = current_val_dice

            torch.save(model.state_dict(), "disease_segmentation_model.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                },
                "checkpoint.pth",
            )
            tqdm.write("---> Best Checkpoint saved.")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            "last_checkpoint.pth",
        )
        tqdm.write("---> Last Checkpoint saved.")

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
