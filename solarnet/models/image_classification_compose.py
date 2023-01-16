import logging
from typing import List, Union
import pytorch_lightning as pl
import pl_bolts.optimizers.lr_scheduler
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from solarnet.models.backbone import get_backbone
from solarnet.models.classifier import Classifier
from torch import nn, optim
from torchmetrics import Accuracy, F1, MetricCollection, Recall, StatScores
from solarnet.utils.metrics import stats_metrics
from solarnet.models.model_utils import BaseModel
import numpy as np
logger = logging.getLogger(__name__)


class ImageClassification_combine(BaseModel):
    """
    Model for image classification.
    This is a configurable class composed by a backbone (see solarnet.models.backbone.py) and
    a classifier.
    It is also a LightningModule and nn.Module.
    """

    def __init__(
        self,

        n_class: int = 2,

        models_dict=None,

        type = 'sum',

        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.models_dict = models_dict
        self.type = type


        self.test_metrics = MetricCollection(
            [
                Accuracy(),
                F1(num_classes=self.hparams.n_class, average="macro"),
                Recall(num_classes=self.hparams.n_class, average="macro"),  # balanced acc.
                StatScores(
                    num_classes=self.hparams.n_class if self.hparams.n_class > 2 else 1,
                    reduce="micro",
                    multiclass=self.hparams.n_class > 2,
                ),
            ]
        )



    @property
    def output_size(self) -> int:
        return self.hparams.n_class


    def forward(self, image):

        # summation
        if self.type == 'sum':
            outs= []
            for channel in self.models_dict.keys():
                self.models_dict[channel].freeze()
                out = F.softmax(self.models_dict[channel](image[channel]), dim = 1)
                outs.append(out)

            out = outs[0]
            for i in range (len(outs)-1):
                out +=outs[i+1]
            out=F.softmax(out, dim=1)

            return out

        if self.type == 'maj':



            #majority voting
        
            outs = []
            count =0
            for channel in self.models_dict.keys():
                self.models_dict[channel].freeze()
                out = F.softmax(self.models_dict[channel](image[channel]), dim=1)
                y_pred = torch.argmax(out, dim=1)
                if count ==0:
                    y_res = y_pred
                else:
                    y_res+=y_pred

                count+=1
            num_maj= int(len(list(self.models_dict.keys()))/2) + 1
            y_res_bin = (y_res>num_maj).int()

            return y_res_bin

        '''
  
        y_pred1 = torch.argmax(out1, dim=1)
        y_pred2=torch.argmax(out2, dim=1)
        #y_pred3=torch.argmax(out3, dim=1)
        y_pred=y_pred1+y_pred2#+y_pred3
        y_pred = (y_pred>1).int()
        b = np.zeros((y_pred.shape[0], y_pred.max() + 1))
        b[np.arange(y_pred.shape[0]), y_pred] = 1
        res = torch.Tensor(b)
        print(res.shape)
        '''




    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        image, _ = batch
        return self(image)



    def test_step(self, batch, batch_idx):
        image, y, _ = batch
        y_pred = self(image)
        #y_pred = F.softmax(y_pred, dim=1)

        self.test_metrics(y_pred, y)

    def test_epoch_end(self, outs):
        test_metrics = self.test_metrics.compute()

        tp, fp, tn, fn, _ = test_metrics.pop("StatScores")
        self.log("test_tp", tp)
        self.log("test_fp", fp)
        self.log("test_tn", tn)
        self.log("test_fn", fn)

        for key, value in test_metrics.items():
            self.log(f"test_{key.lower()}", value)






