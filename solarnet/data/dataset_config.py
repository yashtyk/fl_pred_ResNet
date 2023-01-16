import os
from pathlib import Path
import random
from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torchvision import transforms

from solarnet.data import BaseDataset, SDOBenchmarkDataModule, SDOBenchmarkDataset
from solarnet.data.sdo_benchmark_multi import SDOBenchmarkDataModule_multi, SDOBenchmarkDataset_multi
from solarnet.data.sdo_benchmark_deep import SDOBenchmarkDataModule_deep, SDOBenchmarkDataset_deep
from solarnet.data.transforms import sdo_dataset_normalize



def datamodule_from_config(parameters: dict) -> LightningDataModule:

    from solarnet.utils.target import flux_to_class_builder

    name = parameters["data"]["name"]
    path = Path(parameters["data"]["path"])

    type = parameters["type"] if "type" in parameters.keys() else "one"

    target_transform = flux_to_class_builder(parameters["data"]["targets"]["classes"])

    parameters['seed'] =  random.randint(-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff)


    if name == "sdo-benchmark":
        if type == "one":
            datamodule = SDOBenchmarkDataModule(
                path,
                batch_size=parameters["trainer"]["batch_size"],
                validation_size=parameters["data"]["validation_size"],
                channel=parameters["data"]["channel"],
                resize=parameters["data"]["size"],
                seed=parameters["seed"],
                num_workers=parameters["system"]["workers"],
                target_transform=target_transform,
                time_steps=parameters["data"]["time_steps"],
            )
        if type == "multi":
            datamodule = SDOBenchmarkDataModule_multi(
                path,
                batch_size=parameters["trainer"]["batch_size"],
                validation_size=parameters["data"]["validation_size"],
                channel=parameters["data"]["channel"],
                resize=parameters["data"]["size"],
                seed=parameters["seed"],
                num_workers=parameters["system"]["workers"],
                train = parameters["is_train"],
                #target_transform=target_transform,
                time_steps=parameters["data"]["time_steps"],
            )
        if type == "deep":
            datamodule = SDOBenchmarkDataModule_deep(
                path,
                batch_size=parameters["trainer"]["batch_size"],
                validation_size=parameters["data"]["validation_size"],
                channel=parameters["data"]["channel"],
                resize=parameters["data"]["size"],
                seed=parameters["seed"],
                num_workers=parameters["system"]["workers"],
                target_transform=target_transform,
                time_steps=parameters["data"]["time_steps"],
            )


    else:
        raise ValueError("Dataset not defined")

    return datamodule


def dataset_from_config(params: dict, split: str, transform: Optional[Callable] = None) -> BaseDataset:

    from solarnet.utils.target import flux_to_class_builder

    data = params["data"]

    name = data["name"]
    path = Path(data["path"])

    target_transform =  flux_to_class_builder(data["targets"]["classes"])

    type = params["type"] if "type" in params.keys() else "one"
    if name == "sdo-benchmark":
        if split == "val":
            raise ValueError("val split not supported for this dataset")
        elif split == "train":
            split = "training"

        if type == "one":
            dataset = SDOBenchmarkDataset(
                path / split / "meta_data.csv",
                path / split,
                channel=data["channel"],
                transform=transform,
                target_transform=target_transform,
                time_steps=data["time_steps"],
            )
        if type == "multi":
            dataset = SDOBenchmarkDataset_multi(
                path / split / "meta_data.csv",
                path / split,
                channel=data["channel"],
                transform=transform,
                target_transform=target_transform,
                time_steps=data["time_steps"],
            )
        if type == "deep":
            dataset = SDOBenchmarkDataset_deep(
                path / split / "meta_data.csv",
                path / split,
                channel=data["channel"],
                transform=transform,
                target_transform=target_transform,
                time_steps=data["time_steps"],
            )



    else:
        raise ValueError("Dataset not defined")

    return dataset
