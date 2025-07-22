import json
import os
import torch
from typing import Union
from pathlib import Path
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data_paths: list[str]):
        self.data_paths = data_paths
        
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        return torch.load(self.data_paths[index], weights_only=False)


class MasterDataLoader:
    def __init__(
        self,
        data_dir: str,
        data_split_file: str,
        batch_size: int = 32,
        num_workers: int = 12,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        with open(data_split_file, "r") as fp:
            data_split = json.load(fp)

        self.tr_keys: list = data_split["train"]
        self.vl_keys: list = data_split["validation"]

        self.tr_paths: list = [os.path.join(data_dir, k) + ".pt" for k in self.tr_keys]
        self.vl_paths: list = [os.path.join(data_dir, k) + ".pt" for k in self.vl_keys]

    def tr_dataloader(self, **args):
        tr_dataset = CustomDataset(self.tr_paths, **args)
        tr_dataloader = DataLoader(
            tr_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=_collate_fn,
            **args,
        )
        print(f"Successfully loaded {len(tr_dataset)} train data")

        return tr_dataloader

    def vl_dataloader(self, **args):
        vl_dataset = CustomDataset(self.vl_paths, **args)
        vl_dataloader = DataLoader(
            vl_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
            **args,
        )
        print(f"Successfully loaded {len(vl_dataset)} validation data")

        return vl_dataloader

    # def ts_dataloader(self, **args):


def _collate_fn(batch):
    return Batch.from_data_list(batch)


if __name__ == "__main__":
    mdl = MasterDataLoader(
        data_dir="../data/input",
        data_split_file="../data/tr_vl_split_250717.json",
        batch_size=32,
        num_workers=12,
    )

    tr_dataloader = mdl.tr_dataloader(shuffle=True)
    vl_dataloader = mdl.vl_dataloader(shuffle=False)
