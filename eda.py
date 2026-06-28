import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def run_eda(dataset_path="dataset/plantsegv3/annotations/train", num_classes=116):
    paths = list(Path(dataset_path).glob("*.png"))
    counts = np.zeros(num_classes, dtype=np.int64)
    fg_pixels, bg_pixels = 0, 0

    for p in tqdm(paths, desc="Calculating Exact Distribution"):
        # Load exactly as saved, avoiding PIL's luminance conversion
        mask = np.array(Image.open(p))

        bg_pixels += (mask == 0).sum()
        fg_pixels += (mask > 0).sum()

        bincount = np.bincount(mask.ravel(), minlength=num_classes)
        counts += bincount[:num_classes]

    total = counts.sum()
    print("\n" + "=" * 40)
    print("      DATASET IMBALANCE SUMMARY")
    print("=" * 40)
    print(f"Total Images Scanned : {len(paths)}")
    print(f"Background (Class 0) : {bg_pixels / (bg_pixels + fg_pixels) * 100:.2f}%")
    print(f"Foreground (Disease) : {fg_pixels / (bg_pixels + fg_pixels) * 100:.2f}%")
    print("=" * 40)

    print("\n--- Active Class Breakdown ---")
    for i, c in enumerate(counts):
        if c > 0:
            print(f"Class {i:03d} | {(c / total) * 100:.4f}%")


if __name__ == "__main__":
    run_eda()
