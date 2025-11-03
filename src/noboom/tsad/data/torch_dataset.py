import torch
import numpy as np
import pandas as pd

from .dataset import Dataset


class TorchDataset(Dataset, torch.utils.data.Dataset):
    """

    :param dataset_name: Name of the dataset.
    :param version: Version of the dataset of shape 'major.minor.patch'.
    :param root: Root directory for the dataset.
    :param train: Load the training set if true and the test set if false.
    :param binary_labels: If set, converts labels to traditional binary labels.
    :param download: If set, downloads the data to root.
    :param train_anomalies: If set, include anomalies in the training set.
    :param include_misc_faults: If set, include miscellaneous faults as anomalies.
    :param include_controller_faults:  If set, include controller faults as anomalies.
    """

    def __init__(self, dataset_name: str, version: str, root: str, train: bool = True,
                 binary_labels: bool = False, download: bool = False, train_anomalies: bool = False,
                 include_misc_faults: bool = False, include_controller_faults: bool = False):

        Dataset.__init__(self, dataset_name, version, root, train, binary_labels, download, train_anomalies,
                         include_misc_faults, include_controller_faults)
        torch.utils.data.Dataset.__init__(self)

    def __getitem__(self, item: int) -> tuple[tuple[torch.Tensor], tuple[torch.Tensor]]:
        return (torch.as_tensor(self._samples[item]),), (torch.as_tensor(self._targets[item]),)
