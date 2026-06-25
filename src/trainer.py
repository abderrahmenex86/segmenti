import json
import os

import torch
from torchinfo import summary as torchinfo_summary
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
        self.n_epochs = kwargs.get("epochs", 40)
        self.patience = kwargs.get("patience", 8)
        self.run_dir = kwargs.get("run_dir", "artifacts/run")

        self.best_metric = -1.0
        self.patience_counter = 0
        self.start_epoch = 1

        metric_classes = 2 if self.num_classes == 1 else self.num_classes
        self.dice_metric = DiceScore(num_classes=metric_classes, include_background=False, average="macro").to(device)
        self.iou_metric = MeanIoU(num_classes=metric_classes, include_background=False).to(device)

        self.history = {"train": {"loss": [], "dice": [], "iou": []}, "val": {"loss": [], "dice": [], "iou": []}}

    def train_epoch(self, epoch):
        self.model.train()
        self.dice_metric.reset()
        self.iou_metric.reset()

        total_loss = 0.0
        total_samples = 0

        desc = f"Epoch {epoch:02d}/{self.n_epochs} [Train]"
        pbar = tqdm(self.train_loader, desc=desc, leave=False, unit="batch")

        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            bs = images.size(0)
            total_samples += bs
            total_loss += loss.item() * bs

            with torch.no_grad():
                logits = outputs["out"] if isinstance(outputs, dict) else outputs
                if self.num_classes == 1:
                    classes = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
                else:
                    classes = logits.argmax(dim=1)

                targets_long = masks.squeeze(1).long()
                self.dice_metric.update(classes, targets_long)
                self.iou_metric.update(classes, targets_long)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_dice = self.dice_metric.compute().item() * 100
        epoch_iou = self.iou_metric.compute().item() * 100

        return {"loss": total_loss / total_samples, "dice": epoch_dice, "iou": epoch_iou}

    def evaluate_epoch(self, epoch):
        self.model.eval()
        self.dice_metric.reset()
        self.iou_metric.reset()

        total_loss = 0.0
        total_samples = 0

        desc = f"Epoch {epoch:02d}/{self.n_epochs} [Val]"
        pbar = tqdm(self.val_loader, desc=desc, leave=False, unit="batch")

        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                bs = images.size(0)
                total_samples += bs
                total_loss += loss.item() * bs

                logits = outputs["out"] if isinstance(outputs, dict) else outputs
                if self.num_classes == 1:
                    classes = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
                else:
                    classes = logits.argmax(dim=1)

                targets_long = masks.squeeze(1).long()
                self.dice_metric.update(classes, targets_long)
                self.iou_metric.update(classes, targets_long)

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_dice = self.dice_metric.compute().item() * 100
        epoch_iou = self.iou_metric.compute().item() * 100

        return {"loss": total_loss / total_samples, "dice": epoch_dice, "iou": epoch_iou}

    def save_architecture_txt(self, img_size):
        os.makedirs(self.run_dir, exist_ok=True)
        arch_file = os.path.join(self.run_dir, "architecture.txt")
        try:
            if torchinfo_summary is not None:
                was_training = self.model.training
                self.model.eval()
                with open(arch_file, "w") as f:
                    f.write(str(torchinfo_summary(self.model, input_size=(1, 3, img_size, img_size), verbose=0)))
                if was_training:
                    self.model.train()
            else:
                with open(arch_file, "w") as f:
                    f.write(str(self.model))
        except Exception:
            with open(arch_file, "w") as f:
                f.write(f"Fallback Model String Representation:\n\n{str(self.model)}")

    def save_history(self):
        history_path = os.path.join(self.run_dir, "model_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=4)

    def save_checkpoint(self, epoch, is_best=False):
        os.makedirs(self.run_dir, exist_ok=True)
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "history": self.history,
        }
        torch.save(checkpoint_state, os.path.join(self.run_dir, "checkpoint.pth"))

        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.run_dir, "best_model.pth"))

    def fit(self, img_size):
        self.save_architecture_txt(img_size)
        epoch_pbar = tqdm(range(self.start_epoch, self.n_epochs + 1), desc="Lifecycle Engine", unit="epoch")

        for epoch in epoch_pbar:
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate_epoch(epoch)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["dice"])
                else:
                    self.scheduler.step()

            for k in ["loss", "dice", "iou"]:
                self.history["train"][k].append(train_metrics[k])
                self.history["val"][k].append(val_metrics[k])

            self.save_history()

            current_lr = self.optimizer.param_groups[0]["lr"]
            tqdm.write(
                f"[Epoch {epoch:02d}] LR: {current_lr:.6f} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.2f}%, IoU: {train_metrics['iou']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.2f}%, IoU: {val_metrics['iou']:.2f}%"
            )

            current_metric = val_metrics["dice"]
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
                tqdm.write(
                    f"  --> Best Dice validation metric improved to {self.best_metric:.2f}%. Saving best_model.pth"
                )
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                tqdm.write(f"  --> Metric flat. Patience progression: {self.patience_counter}/{self.patience}")

            self.save_checkpoint(epoch, is_best=False)

            if self.patience_counter >= self.patience:
                tqdm.write(f"Early stopping limit encountered at epoch {epoch}. Terminating run.")
                break

        return self.history

    def resume(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"State file missing: {checkpoint_path}")

        tqdm.write(f"Restoring engine weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint.get("best_metric", -1.0)
        self.history = checkpoint.get("history", self.history)

        tqdm.write(f"Restoration completed. Active training resumes at Epoch {self.start_epoch}")
