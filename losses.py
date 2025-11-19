import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1e-6):
    """
    Dice Loss for binary segmentation.
    pred: raw logits from model
    target: ground truth mask
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


class BCEWithLogitsLossWeighted(nn.Module):
    """
    Weighted Binary Cross Entropy Loss.
    If no weights are provided, it behaves like normal BCEWithLogitsLoss.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, target, weight=None):
        loss = self.bce(logits, target)
        if weight is not None:
            loss = loss * weight
        return loss.mean()


class HybridLoss(nn.Module):
    """
    Combines Dice Loss and Weighted BCE.
    total_loss = Dice + lambda * WeightedBCE
    """
    def __init__(self, lambda_bce=0.5):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.wbce = BCEWithLogitsLossWeighted()

    def forward(self, logits, target, weight=None):
        return dice_loss(logits, target) + self.lambda_bce * self.wbce(logits, target, weight)
