# Author: Jonathan Donzallaz

import logging
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
from pytorch_lightning.callbacks import BackboneFinetuning
from torchvision.transforms import transforms

from solarnet.data.dataset_config import datamodule_from_config
from solarnet.data.transforms import SDOSimCLRDataTransform, sdo_dataset_normalize
from solarnet.models import ImageClassification
from solarnet.models.image_classification_multi_deep import ImageClassification_multi_deep
from solarnet.utils.target import compute_class_weight, flux_to_class_builder
from solarnet.utils.trainer import train

logger = logging.getLogger(__name__)


def model_from_config(parameters: dict, datamodule: LightningDataModule) -> LightningModule:
    steps_per_epoch = len(datamodule.train_dataloader())
    total_steps = parameters["trainer"]["epochs"] * steps_per_epoch

    class_weight = compute_class_weight(datamodule.dataset_train)

    regression = parameters["data"]["targets"] == "regression"

    model = ImageClassification_multi_deep(
        n_channel=datamodule.size(0),
        n_class=len(parameters["data"]["targets"]["classes"]),
        channels_list=parameters["data"]["channel"],
        class_weight=class_weight,
        lr_scheduler_total_steps=total_steps,
        optimizer=parameters["optimizer_type"],
        classifier_type=parameters["type_class"],

        **parameters["model"],


    )

    return model


def train_multi_deep(parameters: dict):
    #seed_everything(parameters["seed"], workers=True)

    datamodule = datamodule_from_config(parameters)
    datamodule.setup()

    model = model_from_config(parameters, datamodule)

    train(parameters, datamodule, model)



