import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks

from src.factory import build_pipeline


def run_smart_inference(image_path=None, run_dir=None, **kwargs):
    assert image_path is not None, "Inference requires a targeting input image path."
    assert run_dir is not None, "Inference requires an explicit targeting run directory."

    run_path = Path(run_dir)
    hyperparameters_file_path = run_path / "hyperparameters.json"
    best_weights_path = run_path / "best_model.pth"

    if not hyperparameters_file_path.exists():
        raise FileNotFoundError(f"Run configuration missing at: {hyperparameters_file_path}")

    historical_hyperparameters = json.loads(hyperparameters_file_path.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    historical_hyperparameters["device"] = device

    model, _, _, _ = build_pipeline(**historical_hyperparameters)

    if not best_weights_path.exists():
        fallback_checkpoint_path = run_path / "checkpoint.pth"
        if fallback_checkpoint_path.exists():
            checkpoint = torch.load(fallback_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("  --> Fallback checkpoint loaded successfully.")
        else:
            raise FileNotFoundError(f"No valid weights located inside run path: {run_path}")
    else:
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        print("  --> Restored optimal weights successfully.")

    model.eval()

    raw_image = Image.open(image_path).convert("RGB")
    image_resolution = int(historical_hyperparameters.get("img_size", 256))
    num_classes = int(historical_hyperparameters.get("num_classes", 116))

    inference_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_resolution, image_resolution), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = inference_transforms(tv_tensors.Image(raw_image))
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        model_outputs = model(input_batch)

    if num_classes == 1:
        predicted_classes = (torch.sigmoid(model_outputs) > 0.5).long().squeeze(0).cpu()
    else:
        predicted_classes = model_outputs.argmax(dim=1).squeeze(0).cpu()

    unique_classes_present = [element.item() for element in torch.unique(predicted_classes) if element.item() > 0]
    print(f"  --> Detected active class indices: {unique_classes_present}")

    channel_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    channel_standard_deviation = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnormalized_image = (input_tensor.cpu() * channel_standard_deviation) + channel_mean
    uint8_image = (unnormalized_image * 255).clamp(0, 255).to(torch.uint8)

    if not unique_classes_present:
        print("  --> Clean prediction. No overlay applied.")
        overlaid_tensor = uint8_image
    else:
        masks = [predicted_classes == target_class for target_class in unique_classes_present]
        stacked_masks = torch.stack(masks)
        overlaid_tensor = draw_segmentation_masks(uint8_image, stacked_masks, alpha=0.6, colors="red")

    plt.figure(figsize=(10, 10))
    plt.imshow(overlaid_tensor.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(f"Segmentation Render | Active Target Classes: {unique_classes_present}")
    plt.tight_layout()

    figures_directory = Path("docs/figs")
    figures_directory.mkdir(parents=True, exist_ok=True)
    output_image_path = figures_directory / f"inference_output_{Path(image_path).name}"
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  --> Exported visual overlay to: {output_image_path}")
