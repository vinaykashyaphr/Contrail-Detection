import torch
import segmentation_models_pytorch as smp


class AdaptiveDiceLoss(smp.losses.DiceLoss):
    def __init__(self, config):
        super(AdaptiveDiceLoss, self).__init__(mode="binary")
        self.alpha = config["alpha-adice"]
        self.beta = config["beta-adice"]

    def forward(self, y_pr, y_gt):
        dice_loss = super(AdaptiveDiceLoss, self).forward(y_pr, y_gt)
        dice_score = 1.0 - dice_loss
        pos_weights = 1.0 / (1.0 + torch.exp(-10.0 * (dice_score - 0.5)))
        adaptive_dice_loss = dice_loss * (pos_weights * self.alpha + self.beta)
        return adaptive_dice_loss


class WeightedDiceLoss(smp.losses.DiceLoss):
    def __init__(self, config):
        super(WeightedDiceLoss, self).__init__(mode="binary")
        self.alpha = config["alpha-dicew"]
        self.beta = config["beta-dicew"]

    def forward(self, y_pred, y_true):
        dice_loss = super(WeightedDiceLoss, self).forward(y_pred, y_true)
        diceW = -1 * (
            self.alpha * torch.pow(dice_loss, self.beta) * torch.log(1 - dice_loss)
        )
        return diceW


class BceDiceLoss(smp.losses.DiceLoss):
    def __init__(self):
        super().__init__()
        self.bce_dice = smp.losses.DiceLoss() + smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        self.bce_dice(y_pred, y_true)
        return super().forward(y_pred, y_true)
