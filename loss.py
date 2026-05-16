import torch.nn as nn
from segmentation_models_pytorch.losses import MULTICLASS_MODE, DiceLoss


class MulticlassDiceLoss(nn.Module):
    def __init__(self, aux_weight=0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(mode=MULTICLASS_MODE, from_logits=True)
        self.aux_weight = aux_weight

    def _compute_loss(self, pred, target):
        return self.ce(pred, target) + self.dice(pred, target)

    def forward(self, outputs, masks):
        masks = masks.squeeze(1).long()

        if isinstance(outputs, dict):
            main_loss = self._compute_loss(outputs["out"], masks)
            aux_loss = self._compute_loss(outputs["aux"], masks)
            return (1 - self.aux_weight) * main_loss + self.aux_weight * aux_loss

        return self._compute_loss(outputs, masks)
