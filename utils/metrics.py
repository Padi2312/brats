import torch


class DiceScore(torch.nn.Module):
    def __init__(
        self,
        num_classes=4,
        eps=1e-7,
        activation=lambda t: torch.softmax(t, dim=1),
        mean=True,
    ):
        super(DiceScore, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.activation = activation
        self.mean = mean 

    def forward(self, logits, targets):
        return dice_score(logits, targets, activation=self.activation,mean=self.mean)


def tp_fp_fn_tn(logits, targets, activation=lambda t: torch.softmax(t, dim=1)):
    probs = logits
    if activation != None:
        probs = activation(logits)

    tp = torch.sum(targets * probs, dim=(2, 3))
    fn = torch.sum((1 - probs) * targets, dim=(2, 3))
    fp = torch.sum((1 - targets) * probs, dim=(2, 3))
    tn = torch.sum((1 - targets) * (1 - probs), dim=(2, 3))

    return tp, fp, fn, tn


def dice_score(
    logits, targets, activation=lambda t: torch.softmax(t, dim=1), eps=1e-7, mean=True
):
    tp, fp, fn, _ = tp_fp_fn_tn(logits, targets, activation)
    dice = (2.0 * tp + eps) / (2.0 * tp + fn + fp + eps)
    if mean:
        return torch.mean(dice)
    else:
        return dice


def IoU(logits, targets, activation=lambda t: torch.softmax(t, dim=1), eps=1e-7):
    probs = logits
    if activation != None:
        probs = activation(logits)

    probs = probs.long()
    targets = targets.long()

    intersection = (probs & targets).sum((1, 2))  # logical AND
    union = (probs | targets).sum((1, 2))  # logical OR
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()
