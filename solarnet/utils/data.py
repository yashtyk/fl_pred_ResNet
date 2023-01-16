# Author: Jonathan Donzallaz



from collections import Counter
from pathlib import Path
import random
from typing import Optional, TypeVar

import pytorch_lightning as pl
import torch
from torch import default_generator
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset

from solarnet.data.dataset_utils import BaseDataset
import random
import numpy as np
from solarnet.utils.plots import plot_image_grid

T = TypeVar("T")

def GetLabels(flux_values):
    labb = []
    for i in flux_values:
        if i > 1e-4:
            labb.append(3)
        elif i > 1e-5:
            labb.append(2)
        elif i > 1e-6:
            labb.append(1)
        else:
            labb.append(0)
    return labb


def GetLabels(flux_values):
    labb = []
    for i in flux_values:
        if i > 1e-4:
            labb.append(3)
        elif i > 1e-5:
            labb.append(2)
        elif i > 1e-6:
            labb.append(1)
        else:
            labb.append(0)
    return labb


def train_test_split(dataset: Dataset[T], test_size: float = 0.1, seed: Optional[int] = None, type='bal'):
    if test_size < 0 or test_size > 1:
        raise ValueError("Test size must be in [0, 1].")

    if type == 'bal':
        print('bal')
        flux_values = dataset.y_1()

        labels = GetLabels(flux_values)

        indx_non = [index for index, element in enumerate(labels) if element == 0]
        indx_c = [index for index, element in enumerate(labels) if element == 1]
        indx_m = [index for index, element in enumerate(labels) if element == 2]
        indx_x = [index for index, element in enumerate(labels) if element == 3]

        if seed is None:
            random.shuffle(indx_non)
            random.shuffle(indx_c)
            random.shuffle(indx_m)
            random.shuffle(indx_x)
        else:
            random.Random(seed).shuffle(indx_non)
            random.Random(seed).shuffle(indx_c)
            random.Random(seed).shuffle(indx_m)
            random.Random(seed).shuffle(indx_x)

        split_n = int(np.floor(len(indx_non) * test_size))
        split_c = int(np.floor(len(indx_c) * test_size))
        split_m = int(np.floor(len(indx_m) * test_size))
        split_x = int(np.floor(len(indx_x) * test_size))

        train_indices_n, val_indices_n = indx_non[split_n:], indx_non[:split_n]
        train_indices_c, val_indices_c = indx_c[split_c:], indx_c[:split_c]
        train_indices_m, val_indices_m = indx_m[split_m:], indx_m[:split_m]
        train_indices_x, val_indices_x = indx_x[split_x:], indx_x[:split_x]

        train_indices = train_indices_n + train_indices_c + train_indices_m + train_indices_x
        val_indices = val_indices_n + val_indices_c + val_indices_m + val_indices_x

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        return [train_subset, val_subset]

    if type == 'split':
        print('split')
        test_size = int(test_size * len(dataset))
        train_size = len(dataset) - test_size

        generator = default_generator if seed is None else torch.Generator().manual_seed(seed)

        return torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

    else:
        print('error_train_test_splitting')








def data_info(datamodule: pl.LightningDataModule, parameters: dict, save_path: Path = None) -> dict:
    """
    Summary of the data used for training/testing.
    Gives: class-balance in each split, shape of the data, range of the data, split sizes, plot of examples.
    Datasets in datamodule must have y() method to access targets (like BaseDataset).
    """

    info = {}

    # Prepare datasets
    datasets = {}
    try:
        datasets["train"] = datamodule.train_dataloader().dataset
    except Exception:
        pass
    try:
        datasets["val"] = datamodule.val_dataloader().dataset
    except Exception:
        pass
    try:
        datasets["test"] = datamodule.test_dataloader().dataset
    except Exception:
        pass

    # Analyze class-balance
    def class_balance(dataset: BaseDataset):
        if isinstance(dataset, BaseDataset):
            y = dataset.y()
        elif isinstance(dataset, Subset):
            ds = dataset.dataset
            if not isinstance(ds, BaseDataset):
                raise AttributeError("dataset must be a BaseDataset or a Subset of a BaseDataset.")
            y = ds.y(dataset.indices)
        else:
            raise AttributeError("dataset must be a BaseDataset or a Subset of a BaseDataset.")

        counter = Counter(y)
        class_names = [list(c.keys())[0] for c in parameters["data"]["targets"]["classes"]]
        return {class_names[i]: counter[i] for i in range(len(class_names))}

    if "data" in parameters and "targets" in parameters["data"] and "classes" in parameters["data"]["targets"]:
        info["class-balance"] = {}
        for ds_name, ds in datasets.items():
            info["class-balance"][ds_name] = class_balance(ds)

    # Shape of data
    info["shape"] = str(datamodule.size())
    '''

    # Range
    if "train" in datasets:
        if isinstance(datasets["train"][0][0], tuple):
            t = torch.cat([datasets["train"][0][0][0], datasets["train"][1][0][0], datasets["train"][2][0][0]])
        else:
            t = torch.cat([datasets["train"][0][0], datasets["train"][1][0], datasets["train"][2][0]])
        info["tensor-data"] = {
            "min": t.min().item(),
            "max": t.max().item(),
            "mean": t.mean().item(),
            "std": t.std().item(),
        }
    
    # Sizes of splits
    info["set-sizes"] = {}
    for ds_name, ds in datasets.items():
        info["set-sizes"][ds_name] = len(ds)

    # Examples of each batch
    def plot_batch(ds_name: str, ds: Dataset, n_images: int = 32):
        if isinstance(ds[0][0], tuple):
            images = [ds[i][0][0][0] for i in random.sample(range(len(ds)), n_images)]
        else:
            images = [ds[i][0][0] for i in random.sample(range(len(ds)), n_images)]

        plot_image_grid(images, max_images=n_images, columns=8, save_path=save_path / f"data-examples-{ds_name}.png")

    if save_path is not None:
        for ds_name, ds in datasets.items():
            plot_batch(ds_name, ds)
    '''

    return info
