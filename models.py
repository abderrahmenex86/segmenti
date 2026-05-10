import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=115):
        super().__init__()
        base_model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )
        self.backbone = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)
        self.classifier_head = nn.Linear(576, num_classes)

    def forward(self, x):
        x = self.dropout(self.flatten(self.pool(self.backbone(x))))
        return self.classifier_head(x)


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
