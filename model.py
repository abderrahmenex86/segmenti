import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)


class DiseaseSegmenter(nn.Module):
    def __init__(self, num_classes=116):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            aux_loss=True,
        )

        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1))

        if self.model.aux_classifier:
            in_channels_aux = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        x = self.model(x)
        if self.training:
            return x
        return x["out"]
