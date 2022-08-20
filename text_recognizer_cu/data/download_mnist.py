from pathlib import Path
import requests


def download_mnist(path):
    url = "https://github.com/pytorch/tutorials/raw/master/_static/"
    filename = "mnist.pkl.gz"

    if not (path / filename).exists():
        content = requests.get(url + filename).content
        (path / filename).open("wb").write(content)

    return path / filename
