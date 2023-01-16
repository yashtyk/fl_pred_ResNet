# Author: Jonathan Donzallaz

import logging
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms

from solarnet.data import datamodule_from_config, dataset_from_config
from solarnet.models.image_classification_multi_deep import ImageClassification_multi_deep
from solarnet.utils.metrics import stats_metrics
from solarnet.utils.plots import plot_confusion_matrix, plot_image_grid,  plot_roc_curve, plot_images
from solarnet.utils.yaml import load_yaml, write_yaml

logger = logging.getLogger(__name__)




def test_multi_deep(parameters: dict, verbose: bool = False):



    model_path = Path(parameters["path"])
    model_class = ImageClassification_multi_deep
    model = model_class.load_from_checkpoint(str(model_path / "model.ckpt"))
    model.parameters()
    model.freeze()


    n_class = len(parameters["data"]["targets"]["classes"])
    parameters["system"]["gpus"] = min(1, parameters["system"]["gpus"])

    datamodule = datamodule_from_config(parameters)
    datamodule.setup("test")


    trainer = pl.Trainer(
        gpus=parameters["system"]["gpus"],
        logger=None,
    )

    raw_metrics = trainer.test(model, datamodule=datamodule, verbose=verbose)
    raw_metrics = raw_metrics[0]

    tp = raw_metrics.pop("test_tp")  # hits
    fp = raw_metrics.pop("test_fp")  # false alarm
    tn = raw_metrics.pop("test_tn")  # correct negative
    fn = raw_metrics.pop("test_fn")  # miss
    print('tp: {}, tn:{}, fp: {}, fn: {}'.format(tp, tn, fp, fn))
    metrics = {"balanced_accuracy": raw_metrics.pop("test_recall"), **stats_metrics(tp, fp, tn, fn)}

    for key, value in raw_metrics.items():
        metrics[key[len("test_"):]] = value
    metrics = dict(sorted(metrics.items()))

    write_yaml(model_path / "metrics.yaml", metrics)


def get_random_test_samples_dataloader(
    parameters: dict,
    nb_sample: int = 10,
    transform: Optional[Callable] = None,
    classes: Optional[List[int]] = None,
) -> Tuple[Dataset, DataLoader]:
    """ Return a random set of test samples """

    dataset_test_image = dataset_from_config(
        parameters, "test", transforms.Compose([transform, transforms.Normalize([-1], [2]), transforms.ToPILImage()])
    )
    dataset_test_tensors = dataset_from_config(parameters, "test", transform)

    if classes is not None:
        y = dataset_test_image.y()
        y = torch.tensor(y)

        # Find number of sample to choose for each class ((nb_sample / len(classes) +/- 1)
        split = (torch.arange(nb_sample + len(classes) - 1, nb_sample - 1, -1) // len(classes)).tolist()
        subset_indices = []
        for i, class_ in enumerate(classes):
            # Indices in the dataset corresponding to this class
            indices_for_class = torch.where(y == class_)[0]
            # Add random indices corresponding to this class
            subset_indices += indices_for_class[torch.randint(len(indices_for_class), (split[i],))].tolist()
    else:
        subset_indices = [random.randrange(len(dataset_test_image)) for _ in range(nb_sample)]

    subset_images = Subset(dataset_test_image, subset_indices)
    subset_tensors = Subset(dataset_test_tensors, subset_indices)
    dataloader_tensors = DataLoader(subset_tensors, batch_size=nb_sample, num_workers=0, shuffle=False)

    return subset_images, dataloader_tensors


def predict(model, dataloader, is_regression: bool = False, return_proba: bool = False):

    y_pred = []
    y_proba = []
    y = []


    with torch.no_grad():
        for i in dataloader:
            logits = model(i[0], i[2])
            y_pred += torch.argmax(logits, dim=1).tolist()
            y_proba += F.softmax(logits, dim=1).tolist()
            y += i[1].tolist()

    if return_proba:
        return y, y_pred, y_proba

    return y, y_pred
