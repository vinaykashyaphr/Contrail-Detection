from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch_optimizer import Lookahead
import torchmetrics
import lightning.pytorch as L
import segmentation_models_pytorch as smp
from losses import AdaptiveDiceLoss, WeightedDiceLoss, BceDiceLoss


class LightningModule(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.apply(self.init_weights)
        self.model = model
        self.config = config
        self.all_dice = []

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()

    def training_step(self, batch, _):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion()(logits, y)
        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, sync_dist=True, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        base = torch.optim.RAdam(self.model.parameters(), lr=self.config["initial-lr"])
        optimizer = Lookahead(optimizer=base)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.config["t0"],
            T_mult=self.config["t-mult"],
            eta_min=self.config["minimum-lr"],
        )
        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.criterion()(y_hat, y)
        self.log(
            "val_loss",
            val_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        indiv_dice = torchmetrics.functional.dice(
            torch.sigmoid(y_hat), y.long()
        ).tolist()
        self.all_dice.append(indiv_dice)

    def schedule_lr(self, scheduler, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def on_validation_epoch_end(self):
        dice_score = torch.mean(torch.tensor(self.all_dice))
        self.log(
            "dice_score",
            dice_score,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.all_dice.clear()

    def criterion(self):
        if self.config["loss_type"] == "adaptive-dice":
            loss_fn = AdaptiveDiceLoss(self.config)
        elif self.config["loss_type"] == "weighted-dice":
            loss_fn = WeightedDiceLoss(self.config)
        elif self.config["loss_type"] == "bce-dice":
            loss_fn = BceDiceLoss()
        else:
            loss_fn = smp.losses.DiceLoss(
                mode="binary", smooth=self.config["loss_smooth"]
            )
        return loss_fn
