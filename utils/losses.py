import torch.nn as nn
from .metrics import dice_score


class DiceLossV4(nn.Module):
    def __init__(self, num_classes=4, eps=1e-7):
        super(DiceLossV4, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        # Apply softmax to logits
        dice = dice_score(logits, targets)
        dice_loss = 1 - dice
        return dice_loss


class DiceCE(nn.Module):
    def __init__(self, ce_weight=1, dice_weight=1):
        super(DiceCE, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLossV4()

    def forward(self, pred, target):
        ce_loss = self.cross_entropy(pred, target)
        d_loss = self.dice_loss(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * d_loss
