import numpy.typing as npt
import numpy as np

from noboom.tsad.metrics import compute_events


def edf(predictions: npt.NDArray[int], targets: npt.NDArray[int]) -> float:
    """Compute the early detection frequency.

    An anomaly is a maximal continuous segment of targets, where targets is not zero.
    An anomaly is detected early, if the first prediction with value 1 in an anomaly window is in the segment where
    targets are 1.

    Example:
    targets     = [0,1,2,3,0,1,2,3,0]
    predictions = [0,1,1,0,0,0,1,1,0]
    Both anomalies are detected, but only the first is detected early.

    The early detection frequency is the fraction of anomalies that are detected early among all detected anomalies.
    EDF = #early detected anomalies / #detected anomalies

    :param predictions: A binary sequence of predictions.
    :param targets: A sequence of ground-truth labels with elements in [0,1,2,3].
    :return: Early Detection Frequency
    """

    # Collect all anomaly events in the ground truth
    anomalies = compute_events(targets)

    # Filter out detected anomalies
    anomalies = [anomaly for anomaly in anomalies if np.sum(predictions[anomaly[0]:anomaly[-1]]) > 0]

    # Compute which anomalies are detected early
    early_detected = [int(np.sum(predictions[anomaly[0]: anomaly[1]]) > 0) for anomaly in anomalies]

    if early_detected:
        return float(np.mean(early_detected))
    else:
        return 0.0
