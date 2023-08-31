import torch
import pandas as pd
import lightning.pytorch as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from sklearn.model_selection import KFold
import gc
import json
from lightning_module import LightningModule
from unet_arch import UnetModule, Unet3Plus, AttentionUnet
from contrails_data import KFoldDataModule
from lightning.pytorch.loops import loop


with open("/content/hparams.json", "r") as hparams:
    config = json.load(hparams)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("/content/paths.json", "r") as hparams:
    paths = json.load(hparams)


class TrainWithKfold(loop._Loop):
    def __init__(self, config, paths) -> None:
        self.config = config
        self.paths = paths
        self.logs_path = "/kaggle/working/"
        self.setup_data()

    def setup_data(self):
        train_df = pd.read_csv(self.paths["training_data"])
        valid_df = pd.read_csv(self.paths["validation_data"])
        train_df["path"] = (
            self.paths["image_data"] + train_df["record_id"].astype(str) + ".npy"
        )
        valid_df["path"] = (
            self.paths["image_data"] + valid_df["record_id"].astype(str) + ".npy"
        )
        self.full_df = pd.concat([train_df, valid_df])
        self.kf = KFold(
            n_splits=self.config["folds"],
            shuffle=True,
            random_state=self.config["seed"],
        )

    def advance(self, fold, train_idx, val_idx):
        print("------ Fold", fold, "------")
        progress_bar = TQDMProgressBar(refresh_rate=1)
        model_checkpoint = ModelCheckpoint(
            save_weights_only=True,
            monitor="dice_score",
            dirpath=self.paths["output_dir"],
            mode="max",
            filename=f"model_{fold}_{{dice_score:.4f}}",
            save_top_k=1,
            verbose=1,
        )

        dataset = KFoldDataModule(self.full_df, self.config, train_idx, val_idx)

        if self.config["unet-variant"] == "attention-unet":
            unet = AttentionUnet()
        elif self.config["unet-variant"] == "unet3+":
            unet = Unet3Plus()
        else:
            unet = UnetModule()

        model = LightningModule(unet, self.config)

        trainer = L.Trainer(
            accelerator=self.config["device"],
            min_epochs=self.config["number_of_epochs"],
            max_epochs=self.config["number_of_epochs"],
            logger=CSVLogger(save_dir=f"{self.logs_path}/logs_f{fold}/"),
            callbacks=[model_checkpoint, progress_bar],
            precision=self.config['precision'],
        )
        
        trainer.fit(model, dataset)

        del (progress_bar, model_checkpoint, dataset, model, trainer)

        torch.cuda.empty_cache()
        gc.collect()

    def run(self):
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.full_df)):
            self.advance(fold, train_idx, val_idx)
