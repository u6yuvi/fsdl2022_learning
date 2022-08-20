import argparse
from typing import Any, Dict,Tuple
import pytorch_lightning as pl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FC1_DIM = 1024
FC2_DIM = 128
FC3_DIM = 128
FC_DROPOUT = 0.5


loss_func = F.cross_entropy

class MLP(pl.LightningModule):
    """Simple MLP suitable for recognizing single characters."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:

        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config

        input_dim = np.prod(self.data_config["input_dims"])
        num_classes = len(self.data_config["mapping"])

        fc1_dim = self.args.get("fc1", FC1_DIM)
        fc2_dim = self.args.get("fc2", FC2_DIM)
        fc3_dim = self.args.get("fc3",FC3_DIM)
        dropout_p = self.args.get("fc_dropout", FC_DROPOUT)

        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.batch_norm1 = nn.BatchNorm1d(fc1_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc3_dim, num_classes)
        self.skip = nn.Identity()

    def forward(self, x):
        x = torch.flatten(x, 1)
        res1 = x
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        #[batch_szie,fc1_dim]
        #print(f'Layer 1 Output shape: {x.shape}')
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        #print(f'Layer 2 Output shape: {x.shape}')
        #[batch_szie,fc2_dim]
        x = torch.cat((x ,self.skip(res1)),axis=1)
        #x = self.skip(res1) +x
        #[batch_size,fc2_dim+input_dim]
        #print(f'Shape after concatenation: {x.shape}')
        x = self.fc3(x)
        #[batch_size,num_classes]
        return x

    def training_step(self: pl.LightningModule, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        xs, ys = batch  # unpack the batch
        outs = self(xs)  # apply the model
        loss =   loss_func(outs, ys)  # compute the (squared error) loss
        self.log("train-loss", loss, prog_bar=True)
        return loss

    def validation_step(self: pl.LightningModule, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        xs, ys = batch  # unpack the batch
        outs = self(xs)  # apply the model
        loss =   loss_func(outs, ys)  # compute the (squared error) loss
        self.log("val-loss", loss, prog_bar=True)
        return loss

    def test_step(self: pl.LightningModule, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        xs, ys = batch  # unpack the batch
        outs = self(xs)  # apply the model
        preds = torch.argmax(outs, dim=1)
        accuracy = (preds == ys).float().mean()
        self.log("test-accuracy", accuracy, prog_bar=True)
        return accuracy

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)  # https://fsdl.me/ol-reliable-img
        return optimizer

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--fc1", type=int, default=FC1_DIM)
        parser.add_argument("--fc2", type=int, default=FC2_DIM)
        parser.add_argument("--fc_dropout", type=float, default=FC_DROPOUT)
        return parser
