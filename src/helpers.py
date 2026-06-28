import json
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt


def download_dataset(destination_directory="dataset"):
    root_path = Path(destination_directory)
    root_path.mkdir(parents=True, exist_ok=True)

    zip_destination_path = root_path / "plantseg.zip"
    zenodo_download_url = "https://zenodo.org/records/17719108/files/plantseg.zip?download=1"

    print(f"[INFO] Initializing download from Zenodo: {zenodo_download_url}")
    print(f"[INFO] Saving archive to: {zip_destination_path}")

    try:
        urllib.request.urlretrieve(zenodo_download_url, zip_destination_path)
        print("[SUCCESS] Download completed. Extracting archive contents...")

        with zipfile.ZipFile(zip_destination_path, "r") as zip_reference:
            zip_reference.extractall(root_path)

        print(f"[SUCCESS] Extraction complete. Dataset structured inside: {destination_directory}")
        zip_destination_path.unlink()
    except Exception as error:
        print(f"[ERROR] Failed to download or extract dataset: {error}")


def verify_dataset_structure(dataset_directory="dataset/plantsegv3"):
    target_path = Path(dataset_directory)
    splits = ["train", "val"]

    print(f"[INFO] Inspecting dataset at: {target_path.resolve()}")
    for split in splits:
        images_directory = target_path / "images" / split
        annotations_directory = target_path / "annotations" / split

        if not images_directory.exists() or not annotations_directory.exists():
            print(f"  [WARNING] Split folder missing for: {split}")
            continue

        image_files = sorted(
            [file.name for file in images_directory.glob("*.*") if file.suffix.lower() in [".jpg", ".png"]]
        )
        annotation_files = sorted([file.name for file in annotations_directory.glob("*.png")])

        print(f"  Split '{split}': Located {len(image_files)} source images and {len(annotation_files)} annotations.")

        mismatched_counter = 0
        for image_name in image_files[:20]:
            base_name = Path(image_name).stem
            expected_annotation_name = f"{base_name}.png"
            if expected_annotation_name not in annotation_files:
                mismatched_counter += 1

        if mismatched_counter > 0:
            print(f"  [ERROR] Alignment mismatch in {mismatched_counter} out of the first 20 file check targets.")
        else:
            print("  --> Verification checks successfully completed.")


def generate_analytics_plots():
    artifacts_path = Path("artifacts")
    if not artifacts_path.exists():
        print("[ERROR] No artifacts folder detected.")
        return

    run_directories = sorted(
        [directory for directory in artifacts_path.iterdir() if directory.is_dir() and directory.name != "default"]
    )
    if not run_directories:
        print("[ERROR] No training runs found inside artifacts.")
        return

    latest_run_directory = run_directories[-1]
    history_file_path = latest_run_directory / "model_history.json"

    if not history_file_path.exists():
        print(f"[ERROR] Metric history missing at: {history_file_path}")
        return

    history = json.loads(history_file_path.read_text())

    train_loss = history["train"]["loss"]
    val_loss = history["val"]["loss"]
    train_dice = history["train"]["dice"]
    val_dice = history["val"]["dice"]
    train_iou = history["train"]["iou"]
    val_iou = history["val"]["iou"]

    epoch_indices = list(range(1, len(train_loss) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epoch_indices, train_loss, "o-", color="royalblue", label="Train")
    axes[0, 0].plot(epoch_indices, val_loss, "s--", color="darkorange", label="Val")
    axes[0, 0].set_title("Focal + Dice Loss Convergence")
    axes[0, 0].set_xlabel("Epoch Index")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epoch_indices, train_dice, "o-", color="forestgreen", label="Train")
    axes[0, 1].plot(epoch_indices, val_dice, "s--", color="crimson", label="Val")
    axes[0, 1].set_title("Dice Overlap Percentage")
    axes[0, 1].set_xlabel("Epoch Index")
    axes[0, 1].set_ylabel("Dice Score (%)")
    axes[0, 1].legend()

    axes[1, 0].plot(epoch_indices, train_iou, "o-", color="indigo", label="Train")
    axes[1, 0].plot(epoch_indices, val_iou, "s--", color="darkorchid", label="Val")
    axes[1, 0].set_title("Mean Intersection over Union (IoU)")
    axes[1, 0].set_xlabel("Epoch Index")
    axes[1, 0].set_ylabel("IoU (%)")
    axes[1, 0].legend()

    generalization_delta = [validation - training for training, validation in zip(train_loss, val_loss)]
    axes[1, 1].bar(epoch_indices, generalization_delta, color="teal", alpha=0.7, label="Val - Train Loss")
    axes[1, 1].set_title("Generalization Delta Variance")
    axes[1, 1].set_xlabel("Epoch Index")
    axes[1, 1].set_ylabel("Loss Variance")
    axes[1, 1].legend()

    plt.tight_layout()
    output_figure_directory = Path("docs/figs")
    output_figure_directory.mkdir(parents=True, exist_ok=True)
    output_figure_path = output_figure_directory / "training_metrics_grid.png"
    plt.savefig(output_figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Diagnostic metric plots generated at: {output_figure_path}")
