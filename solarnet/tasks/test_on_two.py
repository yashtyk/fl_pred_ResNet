import collections
import logging
import random
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms

from solarnet.data import datamodule_from_config, dataset_from_config
from solarnet.models import ImageClassification
from solarnet.models.image_classification_compose import ImageClassification_combine
from solarnet.utils.metrics import stats_metrics
from solarnet.utils.plots import plot_confusion_matrix, plot_image_grid,  plot_roc_curve, \
    plot_test_curve
from solarnet.utils.yaml import load_yaml, write_yaml

from torchmetrics import Accuracy, F1, MetricCollection, Recall, StatScores

logger = logging.getLogger(__name__)

def draw(parameters: dict):


    datamodule = datamodule_from_config(parameters)
    datamodule.setup('test')
    count = 0
    with PdfPages('./samples_exmpl_test.pdf') as pdf:
        for i, j, path in datamodule.dataset_test:
            plt.figure()
            plt.imshow(i[0])
            plt.title(str(path[0]))
            pdf.savefig()
            plt.close()
            print(count)
            count+=1
            if count > 3000:
                break





print('ok')









def GetLabels(labels):
    labb = []
    mult_factor = 1
    for i in labels:
        if i > 1e-4 *mult_factor:
            labb.append(3)
        elif i > 1e-5 * mult_factor:
            labb.append(2)
        elif i > 1e-6 * mult_factor:
            labb.append(1)
        else:
            labb.append(0)
    return labb


def test_new(parameters: dict, verbose: bool = False):
    models_dict = {}
    path_channel = parameters['channel_path']
    channel = list(path_channel.keys())[0]
    path_metrics_save = Path(parameters["path{}".format(path_channel[channel])])


    for channel in path_channel.keys():

        model_path = Path(parameters["path{}".format(path_channel[channel])])
        model_class = ImageClassification
        model = model_class.load_from_checkpoint(str(model_path / "model.ckpt"))
        model.freeze()
        models_dict[channel] = model

    n_class = len(parameters["data"]["targets"]["classes"])
    parameters["system"]["gpus"] = min(1, parameters["system"]["gpus"])
    type = parameters["model"]["type"]

    datamodule = datamodule_from_config(parameters)
    datamodule.setup()

    model_new = ImageClassification_combine(n_class, models_dict, type)
    model_new.freeze()

    trainer = pl.Trainer(
        gpus=parameters["system"]["gpus"],
        logger=None,
    )





    raw_metrics = trainer.test(model_new, datamodule=datamodule, verbose=verbose)
    raw_metrics = raw_metrics[0]

    tp = raw_metrics.pop("test_tp")  # hits
    fp = raw_metrics.pop("test_fp")  # false alarm
    tn = raw_metrics.pop("test_tn")  # correct negative
    fn = raw_metrics.pop("test_fn")  # miss
    print('tp: {}, tn:{}, fp: {}, fn: {}'.format(tp, tn, fp, fn))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy= (tp + tn) / (tp + tn + fp + fn)

    metrics = {"accuracy: ": accuracy, "precision": precision, "recall": recall, "balanced_accuracy": raw_metrics.pop("test_recall"), **stats_metrics(tp, fp, tn, fn)}

    write_yaml(path_metrics_save / 'metrics_update_sum_magn_94_193_211_cont_1700_171.yaml', metrics)
    y = []
    y_proba = []
    y_pred = []
    y_flux= []
    dataloader = datamodule.val_dataloader()
    '''
    # test and plot misclassified examples
    with PdfPages('./miss_74_LAB2.pdf') as pdf:
        with torch.no_grad():
            for idx, i  in enumerate(dataloader):
                logits = model_new(i[0])
                y_pred += torch.argmax(logits, dim=1).tolist()
                y_proba += F.softmax(logits, dim=1).tolist()
                y += i[1].tolist()
                y_flux+=i[2].tolist()
                if not y[-1] == y_pred[-1]:
                    plt.figure()
                    plt.imshow(i[0]['magnetogram'].squeeze()[0], vmin=-1, vmax = 1)
                    #plt.title(str(y_flux[-1]) + ' pred: ' + str(y_pred[-1]) + ' true: '+ str(y[-1]) + str[i[3]])
                    plt.title(i[3])
                    pdf.savefig()
                    plt.close()
                    
                    plt.figure()
                    plt.imshow(datamodule.dataset_val[idx][0]['magnetogram'].squeeze()[0], vmin=-1, vmax=1)
                    plt.title(str(y_flux[-1]) + ' pred: ' + str(y_pred[-1]) + ' true: ' + str(y[-1]))
                    pdf.savefig()
                    plt.close()
                    





   

    types= GetLabels(y_flux)
    idx_non= [idx for idx , x in enumerate(types) if x == 0]
    idx_c = [idx for idx, x in enumerate(types) if x == 1]
    idx_m = [idx for idx, x in enumerate(types) if x == 2]
    idx_x = [idx for idx, x in enumerate(types) if x == 3]

    #check the accuracy for all types

    true_non = [idx for idx, x in enumerate(idx_non) if y_pred[x] == y[x]]
    true_c = [idx for idx, x in enumerate(idx_c) if y_pred[x] == y[x]]
    true_m = [idx for idx, x in enumerate(idx_m) if y_pred[x] == y[x]]
    true_x = [idx for idx, x in enumerate(idx_x) if y_pred[x] == y[x]]

    acc_non = len(true_non) / len(idx_non)
    acc_c = len(true_c) / len(idx_c)
    acc_m = len(true_m) / len(idx_m)
    acc_x = len(true_x) / len(idx_x)

    #check amount of truly classified with probbility >0.75
    true_non_conf = [idx for idx , x in enumerate (idx_non) if y_proba[x][0] >= 0.7]
    true_c_conf = [idx for idx, x in enumerate(idx_c) if y_proba[x][1] >= 0.7]
    true_m_conf = [idx for idx, x in enumerate(idx_m) if y_proba[x][1] >= 0.7]
    true_x_conf = [idx for idx, x in enumerate(idx_x) if y_proba[x][1] >= 0.7]

    print()

    acc = collections.OrderedDict()
    acc['acc_non'] = acc_non
    acc['all_non'] = len(idx_non)
    acc['correct_non'] = len(true_non)
    acc['confident_correct_non'] = len(true_non_conf)
    acc['acc_non_confident'] =  len(true_non_conf) / len(idx_non)



    acc['acc_c'] = acc_c
    acc['all_c'] = len(idx_c)
    acc['correct_c'] = len(true_c)
    acc['confident_correct_c'] = len(true_c_conf)
    acc['acc_c_confident'] = len(true_c_conf) / len(idx_c)

    acc['acc_m'] = acc_m
    acc['all_m'] = len(idx_m)
    acc['correct_m'] = len(true_m)
    acc['confident_correct_m'] = len(true_m_conf)
    acc['acc_m_confident'] = len(true_m_conf) / len(idx_m)

    acc['acc_x'] = acc_x
    acc['all_x'] = len(idx_x)
    acc['correct_x'] = len(true_x)
    acc['confident_correct_x'] = len(true_x_conf)
    acc['acc_x_confident'] = len(true_x_conf) / len(idx_x)

    acc['acc_non_all'] = '{}/{}/{}'.format(acc_non, len(true_non_conf) / len(idx_non), 1)
    acc['val_non_all'] = '{}/{}/{}'.format(len(true_non), len(true_non_conf), len(idx_non))

    acc['acc_c_all'] = '{}/{}/{}'.format(acc_c, len(true_c_conf) / len(idx_c), 1)
    acc['val_c_all'] = '{}/{}/{}'.format(len(true_c), len(true_c_conf), len(idx_c))

    acc['acc_m_all'] = '{}/{}/{}'.format(acc_m, len(true_m_conf) / len(idx_m), 1)
    acc['val_m_all'] = '{}/{}/{}'.format(len(true_m), len(true_m_conf), len(idx_m))

    acc['acc_x_all'] = '{}/{}/{}'.format(acc_x, len(true_x_conf) / len(idx_x), 1)
    acc['val_x_all'] = '{}/{}/{}'.format(len(true_x), len(true_x_conf), len(idx_x))




    name1 = 'type_acc.yaml'

    write_yaml(path_metrics_save / name1, acc)



    '''

    for key, value in raw_metrics.items():
        metrics[key[len("test_"):]] = value
    metrics = dict(sorted(metrics.items()))
    name = str(type)+"_metrics"
    for channel in path_channel.keys():
        name+="_"
        name+=str(channel)

    name+=".yaml"


    write_yaml(path_metrics_save / name, metrics)


def test_on_test(parameters: dict, verbose: bool = False):
    tss = []
    model_path = Path(parameters["path"])
    plot_path = Path(parameters["path"]) / "test_plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    labels = [list(x.keys())[0] for x in parameters["data"]["targets"]["classes"]]
    n_class = len(parameters["data"]["targets"]["classes"])
    parameters["system"]["gpus"] = min(1, parameters["system"]["gpus"])

    datamodule = datamodule_from_config(parameters)
    datamodule.setup("test")
    metadata_path = model_path / "metadata.yaml"
    metr_path = model_path / "metrics.yaml"
    metadata = load_yaml(metadata_path) if metadata_path.exists() else None
    metr = load_yaml(metr_path) if metr_path.exists() else None
    model_class = ImageClassification
    print(metadata['early_stopping_epoch'])
    for i in range(metadata['model_checkpoint_epoch']):
        epoch = i
        step = 52 + i * 53
        name = 'epoch={}-step={}.ckpt'.format(epoch, step)
        model = model_class.load_from_checkpoint(str(model_path / name))

        trainer = pl.Trainer(
            gpus=parameters["system"]["gpus"],
            logger=None,
        )

        # Evaluate model
        raw_metrics = trainer.test(model, datamodule=datamodule, verbose=verbose)
        raw_metrics = raw_metrics[0]

        tp = raw_metrics.pop("test_tp")  # hits
        fp = raw_metrics.pop("test_fp")  # false alarm
        tn = raw_metrics.pop("test_tn")  # correct negative
        fn = raw_metrics.pop("test_fn")  # miss

        metrics = {"balanced_accuracy": raw_metrics.pop("test_recall"), **stats_metrics(tp, fp, tn, fn)}

        for key, value in raw_metrics.items():
            metrics[key[len("test_"):]] = value
        metrics = dict(sorted(metrics.items()))
        print(metrics.keys())
        tss.append(metrics['tss'])
    plot_test_curve(tss, plot_path, metadata['model_checkpoint_epoch'], metr['tss'])




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
    if is_regression:
        y_pred = torch.tensor([])
        y = torch.tensor([])

        with torch.no_grad():
            for i in dataloader:
                y_pred = torch.cat((y_pred, model(i[0]).cpu().flatten()))
                y = torch.cat((y, i[1].cpu().flatten()))

        y = log_min_max_inverse_scale(y)
        y_pred = log_min_max_inverse_scale(y_pred)

        if return_proba:
            return y.tolist(), y_pred.tolist(), None

        return y.tolist(), y_pred.tolist()

    y_pred = []
    y_proba = []
    y = []

    with torch.no_grad():
        for i in dataloader:
            logits = model(i[0])
            y_pred += torch.argmax(logits, dim=1).tolist()
            y_proba += F.softmax(logits, dim=1).tolist()
            y += i[1].tolist()

    if return_proba:
        return y, y_pred, y_proba

    return y, y_pred


def test_on_one(parameters: dict, verbose=False):
    model_path = Path(parameters["path"])
    plot_path = Path(parameters["path"]) / "test_plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    metadata_path = model_path / "metadata.yaml"

    regression = parameters["data"]["targets"] == "regression"
    labels = None if regression else [list(x.keys())[0] for x in parameters["data"]["targets"]["classes"]]
    n_class = 1 if regression else len(parameters["data"]["targets"]["classes"])
    parameters["system"]["gpus"] = min(1, parameters["system"]["gpus"])

    datamodule = datamodule_from_config(parameters)
    datamodule.setup("test")
    logger.info(f"Data format: {datamodule.size()}")

    model_class = ImageRegression if regression else ImageClassification
    model = model_class.load_from_checkpoint(str(model_path / "model.ckpt"))

    predict_one(model, datamodule.test_dataloader())


def predict_one(model, dataloader):
    test_metrics = MetricCollection(
        [
            Accuracy(),
            F1(num_classes=2, average="macro"),
            Recall(num_classes=2, average="macro"),  # balanced acc.
            StatScores(
                num_classes=1,
                reduce="micro",
                multiclass=False,
            ),
        ]
    )

    with torch.no_grad():
        for i in dataloader:
            image, y = i
            y_pred = model(image)
            y_pred = F.softmax(y_pred, dim=1)

            test_metrics(y_pred, y)

        test_metrics = test_metrics.compute()
        tp1, fp1, tn1, fn1, _ = test_metrics.pop("StatScores")

        metrics1 = {**stats_metrics(tp1, fp1, tn1, fn1)}
        print('model1_tss: {}'.format(metrics1['tss']))

        print('tp:{}, tn: {}, fp: {}, fn: {}'.format(tp1, fp1, tn1, fn1))


def predict_combine(model1, model2, dataloader):
    y_pred1_all = []
    y_proba1_all = []
    y = []
    y_proba = []
    y_pred = []
    y_pred2_all = []
    y_proba2_all = []

    test_metrics = MetricCollection(
        [
            Accuracy(),
            F1(num_classes=2, average="macro"),
            Recall(num_classes=2, average="macro"),  # balanced acc.
            StatScores(
                num_classes=1,
                reduce="micro",
                multiclass=False,
            ),
        ]
    )

    test_metrics1 = MetricCollection(
        [
            Accuracy(),
            F1(num_classes=2, average="macro"),
            Recall(num_classes=2, average="macro"),  # balanced acc.
            StatScores(
                num_classes=1,
                reduce="micro",
                multiclass=False,
            ),
        ]
    )
    '''
    test_metrics2 = MetricCollection(
        [
            Accuracy(),
            F1(num_classes=2, average="macro"),
            Recall(num_classes=2, average="macro"),  # balanced acc.
            StatScores(
                num_classes=1,
                reduce="micro",
                multiclass=False,
            ),
        ]
    )
    '''
    with torch.no_grad():
        for i in dataloader:
            image, y = i
            y_pred = model1(image['magnetogram'])
            y_pred = F.softmax(y_pred, dim=1)

            test_metrics(y_pred, y)

            y_pred1 = model2(image[211])
            y_pred1 = F.softmax(y_pred1, dim=1)

            test_metrics1(y_pred1, y)

            #y_predd = y_pred + y_pred1
            #y_predd = F.softmax(y_predd, dim=1)
            #test_metrics2(y_predd, y)

        test_metrics = test_metrics.compute()
        tp1, fp1, tn1, fn1, _ = test_metrics.pop("StatScores")

        test_metrics1 = test_metrics1.compute()
        tp2, fp2, tn2, fn2, _ = test_metrics1.pop("StatScores")

        #test_metrics2 = test_metrics2.compute()
        #tp3, fp3, tn3, fn3, _ = test_metrics2.pop("StatScores")

        metrics1 = {**stats_metrics(tp1, fp1, tn1, fn1)}
        print('model1_tss: {}'.format(metrics1['tss']))

        metrics2 = {**stats_metrics(tp2, fp2, tn2, fn2)}
        print('model2_tss:{}'.format(metrics2['tss']))
        '''
        metrics3 = {**stats_metrics(tp3, fp3, tn3, fn3)}
        print('combine_tss: {}'.format(metrics3['tss']))
        '''

