from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch
import lightning.pytorch as L
import pandas as pd


class ContrailData(Dataset):
    def __init__(self, df, image_size=256, train=True):
        self.df = df
        self.trn = train
        self.normalize_image = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
        self.image_size = image_size
        if image_size != 256:
            self.resize_image = transforms.transforms.Resize(image_size)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        con_path = row.path
        con = np.load(str(con_path))

        img = con[..., :-1]
        label = con[..., -1]

        label = torch.tensor(label)

        img = (
            torch.from_numpy(np.reshape(img, (256, 256, 3)))
            .to(torch.float32)
            .permute(2, 0, 1)
        )

        if self.image_size != 256:
            img = self.resize_image(img)

        img = self.normalize_image(img)

        return img.float(), label.float()

    def __len__(self):
        return len(self.df)


class ContrailLightData(L.LightningDataModule):
    def __init__(self, config, paths):
        super().__init__()
        self.config = config
        self.paths = paths

    def setup(self, stage):
        train_df = pd.read_csv(self.paths["training_data"])
        valid_df = pd.read_csv(self.paths["validation_data"])
        train_df["path"] = (
            self.paths["image_data"] + train_df["record_id"].astype(str) + ".npy"
        )
        valid_df["path"] = (
            self.paths["image_data"] + valid_df["record_id"].astype(str) + ".npy"
        )
        self.train_dataset = ContrailData(train_df)
        self.valid_dataset = ContrailData(valid_df)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config["train_batch_size"],
            num_workers=self.config["num_workers"],
            drop_last=self.config['drop-last-row'],
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.config["valid_batch_size"],
            num_workers=self.config["num_workers"],
            drop_last=self.config['drop-last-row'],
        )


class KFoldDataModule(L.LightningDataModule):
    def __init__(self, df, config, train_idx, val_idx):
        super().__init__()
        self.full_df = df
        self.config = config
        self.train_indexes, self.val_indexes = train_idx, val_idx

    def setup(self, stage=None):
        data_train_df, data_val_df = (
            self.full_df.iloc[self.train_indexes],
            self.full_df.iloc[self.val_indexes],
        )
        self.data_train = ContrailData(data_train_df)
        self.data_val = ContrailData(data_val_df)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.config["train_batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
            drop_last=self.config['drop-last-row'],
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.config["valid_batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            drop_last=self.config['drop-last-row'],
        )
