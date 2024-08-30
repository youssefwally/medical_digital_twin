from typing import List

import torch
from torch import optim
from torch import nn
from rga import lr_schedulers

from rga.metrics.edge_accuracy import EdgeAccuracy, MaskAccuracy
from rga.metrics.graph_size import MaxGraphSize
from rga.metrics.losses import (
    MeanEmbeddingsLoss,
    MeanKLDLoss,
    MeanReconstructionLoss,
    MeanClassificationLoss,
)
from rga.metrics.graph_drawer import GraphDrawer
from rga.metrics.precision_recall import *


def get_loss(name: str, loss_weight: torch.Tensor = None):
    losses = {
        "MSE": torch.nn.MSELoss,
        "BCE": torch.nn.BCELoss,
        "BCEWithLogits": torch.nn.BCEWithLogitsLoss,
        "CrossEntropy": nn.CrossEntropyLoss,
    }
    if loss_weight is not None:
        if isinstance(loss_weight, float):
            loss_weight = torch.Tensor([loss_weight])
        return losses[name](weight=loss_weight)
    else:
        return losses[name]()


def get_optimizer(name: str):
    optimizers = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "AdamWAMSGrad": AdamWAMSGrad,
        "SGD": optim.SGD,
    }
    return optimizers[name]


def AdamWAMSGrad(*args, **kwargs):
    return optim.AdamW(amsgrad=True, *args, **kwargs)


def get_lr_scheduler(name: str):
    if name is None:
        return lr_schedulers.NoSched
    optimizers = {
        "none": lr_schedulers.NoSched,
        "NoSched": lr_schedulers.NoSched,
        "FactorDecreasingOnMetricChange": lr_schedulers.FactorDecreasingOnMetricChange,
        "factor_decreasing_on_metric_change": lr_schedulers.FactorDecreasingOnMetricChange,
        "SingleTimeChangeOnMetricTreshold": lr_schedulers.SingleTimeChangeOnMetricTreshold,
        "StepLR": torch.optim.lr_scheduler.StepLR,
        "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }
    return optimizers[name]


def get_metrics(metrics: List[str]):
    if metrics is None:
        return []
    metrics_dict = {
        "EdgeAccuracy": EdgeAccuracy,
        "MaskAccuracy": MaskAccuracy,
        "EdgePrecision": EdgePrecision,
        "EdgeRecall": EdgeRecall,
        "EdgePrecisionNonWeighted": EdgePrecisionNonWeighted,
        "EdgeRecallNonWeighted": EdgeRecallNonWeighted,
        "EdgePrecisionSquareWeighted": EdgePrecisionSquareWeighted,
        "EdgeRecallSquareWeighted": EdgeRecallSquareWeighted,
        "EdgeF1": EdgeF1,
        "MaskPrecision": MaskPrecision,
        "MaskRecall": MaskRecall,
        "MaxGraphSize": MaxGraphSize,
        "MeanReconstructionLoss": MeanReconstructionLoss,
        "MeanEmbeddingsLoss": MeanEmbeddingsLoss,
        "MeanClassificationLoss": MeanClassificationLoss,
        "MeanKLDLoss": MeanKLDLoss,
        "GraphDrawer": GraphDrawer,
        "Accuracy": torchmetrics.Accuracy,
        "F1": torchmetrics.F1,
        "Precision": torchmetrics.Precision,
        "Recall": torchmetrics.Recall,
    }
    return [metrics_dict[m]() for m in metrics]


def get_activation_function(name: str = "ReLU"):
    d = {
        "ReLU": nn.ReLU,
        "ELU": nn.ELU,
        "CELU": nn.CELU,
    }
    return d[name]
