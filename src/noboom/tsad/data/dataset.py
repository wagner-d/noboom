import os
import numpy as np
import pandas as pd


class Dataset:
    """Simple dataset class that loads all sequences of a dataset into a list of numpy arrays.

    :param dataset: Name of the dataset.
    :param version: Version of the dataset of shape 'major.minor.patch'.
    :param root: Root directory for the dataset.
    :param train: Load the training set if true and the test set if false.
    :param binary_labels: If set, converts labels to traditional binary labels.
    :param download: If set, downloads the data to root.
    :param train_anomalies: If set, include anomalies in the training set.
    :param include_misc_faults: If set, include miscellaneous faults as anomalies.
    :param include_controller_faults:  If set, include controller faults as anomalies.
    """

    def __init__(self, dataset: str, version: str, root: str, train: bool = True,
                 binary_labels: bool = False, download: bool = False, train_anomalies: bool = False,
                 include_misc_faults: bool = False, include_controller_faults: bool = False):

        self.name = dataset
        self.version = version
        self.root = root
        self.train = train
        self.binary_labels = binary_labels
        self.train_anomalies = train_anomalies
        self.include_misc_faults = include_misc_faults
        self.include_controller_faults = include_controller_faults

        self._features: list[str] = []
        self._samples = []
        self._targets = []

        self.load()

    def load(self):

        if self.train:
            if self.train_anomalies:
                prefix = 'train'
            else:
                prefix = 'train_normal'
        else:
            prefix = 'test'

        for (operating_point, _, runs) in os.walk(self.root):

            for run in runs:
                if prefix in run and run.endswith('.csv'):

                    time_series = pd.read_csv(os.path.join(operating_point, run))

                    if self.include_misc_faults and self.label_misc_feature is not None:

                        mask = (time_series[self.label_misc_feature] > 0) & (time_series[self.label_feature] == 0)
                        time_series.loc[mask, self.label_feature] = time_series.loc[mask, self.label_misc_feature]

                    if self.include_controller_faults and self.label_controller_feature is not None:

                        mask = (time_series[self.label_controller_feature] > 0) & (time_series[self.label_feature] == 0)
                        time_series.loc[mask, self.label_feature] = time_series.loc[mask, self.label_controller_feature]

                    if self.binary_labels:
                        targets = (time_series[self.label_feature].to_numpy() > 0).astype(int)
                    else:
                        targets = time_series[self.label_feature].to_numpy()

                    time_series.drop(self.meta_data, axis=1, inplace=True)

                    time_series = time_series.astype(np.float32)

                    self._samples.append(time_series.to_numpy())
                    self._targets.append(targets)

                    del time_series

    def __getitem__(self, item) -> tuple[np.ndarray, np.ndarray]:
        return self._samples[item], self._targets[item]

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def seq_len(self) -> list[int]:
        return [ts.data.shape[0] for ts in self._samples]

    @property
    def num_features(self) -> int:
        return self._samples[0].shape[-1]

    @property
    def label_feature(self):
        return 'Anomaly' if self.name == 'industry_process' else 'Label (advanced/hard fault)'

    @property
    def label_misc_feature(self):
        return None if self.name == 'industry_process' else 'Label (advanced/soft fault)'

    @property
    def label_controller_feature(self):
        return None if self.name == 'industry_process' else 'Label (advanced/controller fault)'

    @property
    def meta_data(self):
        return ['Anomaly'] if self.name == 'industry_process' else ['Time', 'Label (common/hard fault)',
                                                                    'Label (common/soft fault)',
                                                                    'Label (common/controller fault)',
                                                                    'Label (common/hard and soft)',
                                                                    'Label (common/all)',
                                                                    'Label (advanced/hard fault)',
                                                                    'Label (advanced/soft fault)',
                                                                    'Label (advanced/controller fault)']