import argparse
import datetime
import json
import os

import torch

from src.dataset import get_dataloaders
from src.factory import build_criterion, build_model, build_optimizer, build_scheduler
from src.infer import run_smart_inference
from src.optimize import run_optuna_study
from src.tester import run_sanity_checks
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Segmenti Crop Disease Segmentation Lifecycle Engine")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["test", "train", "optimize", "infer"], help="Active engine phase"
    )

    parser.add_argument("--root_dir", type=str, default="dataset/plantsegv3", help="Target dataset folder path")
    parser.add_argument("--batch_size", type=int, default=10, help="DataLoader batch size")
    parser.add_argument("--img_size", type=int, default=520, help="Resolution sizing parameter")
    parser.add_argument("--num_classes", type=int, default=116, help="Target classification mapping count")

    parser.add_argument(
        "--model_type", type=str, default="unet", choices=["unet", "deeplabv3"], help="Active model architecture"
    )
    parser.add_argument("--encoder_name", type=str, default="mobilenet_v2", help="Feature extractor backbone")
    parser.add_argument("--encoder_weights", type=str, default="imagenet", help="Source pretraining state")

    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="L2 penalty coefficient")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"], help="Numerical optimizer selection"
    )
    parser.add_argument("--pos_weight", type=float, default=5.0, help="Positive binary category weighting scale")

    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "step", "none"],
        help="LR decay sequence builder",
    )

    parser.add_argument("--epochs", type=int, default=40, help="Training sweeps limit")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience threshold")
    parser.add_argument("--resume", action="store_true", help="Flag to recover and continue training")
    parser.add_argument("--optuna_trials", type=int, default=15, help="Optuna hyperparameter sweep scale")

    parser.add_argument("--image_path", type=str, default=None, help="Inference image source path")
    parser.add_argument("--run_dir", type=str, default=None, help="Target historical folder directory")

    parser.add_argument(
        "--class_mapping",
        type=str,
        default=None,
        help="Path to JSON file containing name/index pairs for class names conversion",
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("artifacts", exist_ok=True)

    if args.run_dir is None:
        if args.mode in ["train", "optimize"]:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_dir = os.path.join("artifacts", f"{now}_{args.model_type}_{args.num_classes}class")
        else:
            args.run_dir = "artifacts/default_run"

    checkpoint_to_load = None
    if args.resume and args.mode == "train":
        runs = sorted(
            [
                os.path.join("artifacts", d)
                for d in os.listdir("artifacts")
                if os.path.isdir(os.path.join("artifacts", d)) and not d.startswith("default_")
            ]
        )
        if runs:
            args.run_dir = runs[-1]
            checkpoint_path = os.path.join(args.run_dir, "checkpoint.pth")
            if os.path.exists(checkpoint_path):
                checkpoint_to_load = checkpoint_path
                print(f"Identified recovery checkpoint at: {checkpoint_path}")
            else:
                print(f"Selected: {args.run_dir}, but checkpoint.pth is missing. Starting afresh.")
        else:
            print("No existing run files located. Starting fresh training session.")

    if args.mode == "train" and checkpoint_to_load is None:
        os.makedirs(args.run_dir, exist_ok=True)
        with open(os.path.join(args.run_dir, "hyperparameters.json"), "w") as f:
            json.dump(vars(args), f, default=str, indent=4)

    if args.mode == "test":
        run_sanity_checks(args)

    elif args.mode == "train":
        print(f"Initializing run for {args.model_type} [Classes: {args.num_classes}]")
        print(f"Output targets folder: {args.run_dir}")

        train_loader, val_loader = get_dataloaders(
            root_dir=args.root_dir, batch_size=args.batch_size, num_classes=args.num_classes, img_size=args.img_size
        )

        model = build_model(
            model_type=args.model_type,
            num_classes=args.num_classes,
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
        ).to(args.device)

        optimizer = build_optimizer(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            model_type=args.model_type,
            optimizer=args.optimizer,
        )

        scheduler = build_scheduler(optimizer, scheduler_type=args.scheduler)
        criterion = build_criterion(args.num_classes, pos_weight=args.pos_weight)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            **vars(args),
        )

        if checkpoint_to_load:
            trainer.resume(checkpoint_to_load)

        trainer.fit(img_size=args.img_size)

    elif args.mode == "optimize":
        print(f"Beginning hyperparameter optimization sweeps. (Target limit: {args.optuna_trials})")
        run_optuna_study(args)

    elif args.mode == "infer":
        if args.image_path is None:
            parser.error("--image_path must be specified when using --mode infer.")
        run_smart_inference(
            image_path=args.image_path, run_dir=args.run_dir, class_mapping_path=args.class_mapping, save_output=True
        )


if __name__ == "__main__":
    main()
