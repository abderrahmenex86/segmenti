import segmentation_models_pytorch.losses as smp_losses
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = smp_losses.DiceLoss(
            mode=smp_losses.BINARY_MODE, from_logits=True
        )

    def forward(self, outputs, masks):
        return self.bce(outputs, masks) + self.dice(outputs, masks)
