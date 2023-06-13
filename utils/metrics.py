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
        return dice_score(logits, targets, activation=self.activation, mean=self.mean)


class JaccardScore(torch.nn.Module):
    def __init__(
        self,
        num_classes=4,
        eps=1e-7,
        activation=lambda t: torch.softmax(t, dim=1),
        mean=True,
    ):
        super(JaccardScore, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.activation = activation
        self.mean = mean

    def forward(self, logits, targets):
        return jaccard_index(logits, targets, activation=self.activation)


class PixelAccuracy(torch.nn.Module):
    def __init__(
        self,
        mean=True,
    ):
        super(PixelAccuracy, self).__init__()
        self.mean = mean

    def forward(self, net, targets):
        return pixel_accuracy(net, targets)


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


def jaccard_index(
    logits, targets, activation=lambda t: torch.softmax(t, dim=1), eps=1e-7, mean=True
):
    tp, fp, fn, _ = tp_fp_fn_tn(logits, targets, activation)
    jaccard = (tp + eps) / (tp + fn + fp + eps)
    if mean:
        return torch.mean(jaccard)
    else:
        return jaccard


def pixel_accuracy(net, targets):
    predicted_labels = torch.argmax(net, dim=1)
    target_labels = torch.argmax(targets, dim=1)
    correct_pixels = torch.sum(predicted_labels == target_labels).item()
    total_pixels = predicted_labels.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy
