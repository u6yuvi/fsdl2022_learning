import pytorch_lightning as pl
import gzip
import torch
import pickle
from .utils import BaseDataset,push_to_device



class MNISTDataModuleLightning(pl.LightningDataModule):
    url = "https://github.com/pytorch/tutorials/raw/master/_static/"
    filename = "mnist.pkl.gz"
    
    def __init__(self, dir, bs=32):
        self.dir = dir
        self.bs = bs
        self.path = self.dir / self.filename
        self.pull_data()
        super().__init__()

    # def prepare_data1(self):
    #     if not (self.path).exists():
    #         content = requests.get(self.url + self.filename).content
    #         self.path.open("wb").write(content)

    def pull_data(self):
        with gzip.open(self.path, "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

        x_train, y_train, x_valid, y_valid = map(
            torch.tensor, (x_train, y_train, x_valid, y_valid)
            )
        
        self.train_ds = BaseDataset(x_train, y_train, transform=push_to_device, target_transform=push_to_device)
        self.valid_ds = BaseDataset(x_valid, y_valid, transform=push_to_device, target_transform=push_to_device)

    def setup(self, stage=None):  # prepares state that needs to be set for each GPU on each node
        if stage == "fit" or stage is None:  # other stages: "test", "predict"
            self.train_dataset = self.train_ds
            self.val_dataset = self.valid_ds

        if stage =='test':
            self.val_dataset = self.valid_ds


    def prepare_data(self):  # prepares state that needs to be set once per node
        pass  # but we don't have any "node-level" computations

    def train_dataloader(self: pl.LightningDataModule) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_ds, batch_size=32)

    def val_dataloader(self: pl.LightningDataModule) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.valid_ds, batch_size=32)

    def test_dataloader(self: pl.LightningDataModule) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.valid_ds, batch_size=32)
