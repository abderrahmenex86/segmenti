import json
from pathlib import Path

import torch
from torchmetrics.segmentation import DiceScore, MeanIoU
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, **kwargs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.num_classes = kwargs.get("num_classes", 116)
        self.epochs = kwargs.get("epochs", 100)
        self.patience = kwargs.get("patience", 10)
        self.run_directory = Path(kwargs.get("run_dir", "artifacts/default"))
        self.clip_gradient_max_norm = kwargs.get("clip_grad_norm", 1.0)

        self.best_validation_dice = -1.0
        self.patience_counter = 0
        self.start_epoch = 1

        self.history = {"train": {"loss": [], "dice": [], "iou": []}, "val": {"loss": [], "dice": [], "iou": []}}

        metric_classes = 2 if self.num_classes == 1 else self.num_classes
        include_background_flag = kwargs.get("include_background", False)

        self.dice_metric = DiceScore(
            num_classes=metric_classes,
            average="macro",
            input_format="index",
            include_background=include_background_flag,
        ).to(device)

        self.iou_metric = MeanIoU(
            num_classes=metric_classes, input_format="index", include_background=include_background_flag
        ).to(device)

        self._serialize_architecture()

    def _serialize_architecture(self):
        self.run_directory.mkdir(parents=True, exist_ok=True)
        try:
            from torchinfo import summary

            architecture_string = str(summary(self.model, input_size=(1, 3, 128, 128), verbose=0))
        except ImportError:
            architecture_string = str(self.model)
        (self.run_directory / "architecture.txt").write_text(architecture_string)

    def _process_predictions_and_targets(self, outputs, targets):
        if self.num_classes == 1:
            predictions = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            processed_targets = targets.squeeze(1).long()
        else:
            predictions = outputs.argmax(dim=1)
            processed_targets = targets.squeeze(1).long()
        return predictions, processed_targets

    def _run_epoch(self, loader, is_training, epoch_index):
        self.model.train() if is_training else self.model.eval()
        self.dice_metric.reset()
        self.iou_metric.reset()

        total_loss = 0.0
        total_samples = 0

        description = f"Epoch {epoch_index:02d}/{self.epochs} [{'Train' if is_training else 'Val'}]"
        progress_bar = tqdm(loader, desc=description, leave=False, unit="batch")

        context_manager = torch.enable_grad() if is_training else torch.no_grad()
        with context_manager:
            for images, masks in progress_bar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                if is_training:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                if is_training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_gradient_max_norm)
                    self.optimizer.step()

                batch_size = images.size(0)
                total_samples += batch_size
                total_loss += loss.item() * batch_size

                predictions, targets = self._process_predictions_and_targets(outputs, masks)
                self.dice_metric.update(predictions, targets)
                self.iou_metric.update(predictions, targets)

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = total_loss / total_samples
        epoch_dice = self.dice_metric.compute().item() * 100
        epoch_iou = self.iou_metric.compute().item() * 100

        return epoch_loss, epoch_dice, epoch_iou

    def fit(self):
        for epoch_index in tqdm(range(self.start_epoch, self.epochs + 1), desc="Engine Lifecycle"):
            train_loss, train_dice, train_iou = self._run_epoch(self.train_loader, True, epoch_index)
            val_loss, val_dice, val_iou = self._run_epoch(self.val_loader, False, epoch_index)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()

            for key, values in zip(
                ["loss", "dice", "iou"], [(train_loss, val_loss), (train_dice, val_dice), (train_iou, val_iou)]
            ):
                self.history["train"][key].append(values[0])
                self.history["val"][key].append(values[1])

            (self.run_directory / "model_history.json").write_text(json.dumps(self.history, indent=4))

            tqdm.write(
                f"[Epoch {epoch_index:02d}] "
                f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.2f}% | IoU: {train_iou:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Dice: {val_dice:.2f}% | IoU: {val_iou:.2f}%"
            )

            is_best = val_dice > self.best_validation_dice
            if is_best:
                self.best_validation_dice = val_dice
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.run_directory / "best_model.pth")
                tqdm.write(f"  --> Saved new best model with Validation Dice: {val_dice:.2f}%")
            else:
                self.patience_counter += 1
                tqdm.write(f"  --> Validation performance flat. Patience: {self.patience_counter}/{self.patience}")

            self.save_checkpoint(epoch_index)

            if self.patience_counter >= self.patience:
                tqdm.write("Early stopping limit encountered. Terminating training run.")
                break

        return self.history

    def save_checkpoint(self, epoch_index):
        checkpoint_state = {
            "epoch": epoch_index,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_validation_dice": self.best_validation_dice,
            "history": self.history,
        }
        torch.save(checkpoint_state, self.run_directory / "checkpoint.pth")

    def resume(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_validation_dice = checkpoint.get("best_validation_dice", -1.0)
        self.history = checkpoint.get("history", self.history)
        tqdm.write(f"Resumed successfully. Training starts at epoch {self.start_epoch}")
