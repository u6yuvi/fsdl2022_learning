from text_recognizer_cu.data.mnist import MNISTDataModuleLightning
from text_recognizer_cu.data.download_mnist import download_mnist
from text_recognizer_cu.models.mlp import MLP
import pytorch_lightning as pl
import torch
import argparse
from pathlib import Path

from argparse import Namespace  # you'll need this


def main():
    #download mnist data
    data_path = Path("data") if Path("data").exists() else Path("./data")
    path = data_path / "downloaded" / "vector-mnist"
    path.mkdir(parents=True, exist_ok=True)
    datafile = download_mnist(path)


    #model-data-config
    config = {"fc1":256,"fc2":128,"fc3": 128+784, "fc_dropout":0.1}
    args = Namespace(**config)  # edit this
    digits_to_9 = list(range(10))
    data_config = {"input_dims": (784,), "mapping": {digit: str(digit) for digit in digits_to_9}}
    data_config

    model = MLP(data_config, args=args)

    datamodule = MNISTDataModuleLightning(dir=path)

    trainer = pl.Trainer(max_epochs=5, gpus=int(torch.cuda.is_available()))

    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model,datamodule=datamodule)

if __name__ == "__main__":
    main()