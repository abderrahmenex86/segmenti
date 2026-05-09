import segmentation_models_pytorch as smp
import torch.nn as nn


class DiseaseSegmenter(nn.Module):
    def __init__(self, encoder_name="mobilenet_v2", encoder_weights="imagenet"):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )

    def forward(self, x):
        return self.model(x)
