import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from src.dataset import get_dataloaders
from src.factory import build_pipeline
from src.infer import run_smart_inference
from src.optimize import run_optuna_study
from src.tester import run_sanity_checks
from src.trainer import Trainer


def main():
    argument_parser = argparse.ArgumentParser(description="Segmenti Pure-PyTorch ML Lifecycle Engine")
    argument_parser.add_argument("--mode", required=True, choices=["test", "train", "optimize", "infer"])
    argument_parser.add_argument("--root_dir", default="dataset/plantseg")
    argument_parser.add_argument("--run_dir", default=None)
    argument_parser.add_argument("--resume", action="store_true")
    argument_parser.add_argument("--image_path", default=None)

    parsed_arguments, unknown_arguments = argument_parser.parse_known_args()

    hyperparameters = vars(parsed_arguments)
    hyperparameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unknown_arguments_dictionary = {
        unknown_arguments[index].lstrip("-"): (
            (float(value) if "." in value else int(value)) if value.replace(".", "", 1).isdigit() else value
        )
        for index, value in zip(range(0, len(unknown_arguments), 2), unknown_arguments[1::2])
    }
    hyperparameters.update(unknown_arguments_dictionary)

    hyperparameters.setdefault("num_classes", 116)
    hyperparameters.setdefault("img_size", 256)
    hyperparameters.setdefault("batch_size", 16)
    hyperparameters.setdefault("epochs", 50)
    hyperparameters.setdefault("patience", 5)
    hyperparameters.setdefault("model_type", "unet")
    hyperparameters.setdefault("optimizer_type", "AdamW")
    hyperparameters.setdefault("learning_rate", 1e-3)
    hyperparameters.setdefault("weight_decay", 1e-4)
    hyperparameters.setdefault("scheduler_type", "ReduceLROnPlateau")

    if hyperparameters["run_dir"] is None:
        if parsed_arguments.mode in ["train", "optimize"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = hyperparameters["model_type"]
            class_count = hyperparameters["num_classes"]
            hyperparameters["run_dir"] = f"artifacts/{timestamp}_{model_name}_{class_count}class"
        else:
            hyperparameters["run_dir"] = "artifacts/default"

    run_directory = Path(hyperparameters["run_dir"])
    checkpoint_path = None

    if parsed_arguments.resume and parsed_arguments.mode == "train":
        artifacts_path = Path("artifacts")
        historical_runs = sorted(
            [directory for directory in artifacts_path.glob("*") if directory.is_dir() and directory.name != "default"]
        )
        if historical_runs:
            last_run = historical_runs[-1]
            target_checkpoint = last_run / "checkpoint.pth"
            if target_checkpoint.exists():
                hyperparameters["run_dir"] = str(last_run)
                run_directory = last_run
                checkpoint_path = target_checkpoint
                print(f"[INFO] Resume target identified: {checkpoint_path}")

    if parsed_arguments.mode in ["train", "optimize"] and checkpoint_path is None:
        run_directory.mkdir(parents=True, exist_ok=True)
        serializable_hyperparameters = {
            key: str(value) if isinstance(value, (Path, torch.device)) else value
            for key, value in hyperparameters.items()
        }
        (run_directory / "hyperparameters.json").write_text(json.dumps(serializable_hyperparameters, indent=4))

    if parsed_arguments.mode == "test":
        run_sanity_checks(**hyperparameters)

    elif parsed_arguments.mode == "train":
        train_loader, val_loader = get_dataloaders(**hyperparameters)
        model, criterion, optimizer, scheduler = build_pipeline(**hyperparameters)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **hyperparameters,
        )

        if checkpoint_path is not None:
            trainer.resume(checkpoint_path)

        trainer.fit()

    elif parsed_arguments.mode == "optimize":
        run_optuna_study(**hyperparameters)

    elif parsed_arguments.mode == "infer":
        run_smart_inference(**hyperparameters)


if __name__ == "__main__":
    main()
