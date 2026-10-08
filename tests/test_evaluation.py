import numpy as np

from src.evaluation.calibration import ece_binary, macro_ece, sigmoid
from src.evaluation.external_validation import LABELS


def test_calibration_helpers():
    y = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    p = np.array([[0.1, 0.9], [0.8, 0.7], [0.2, 0.2], [0.9, 0.1]])
    assert 0 <= ece_binary(y[:, 0], p[:, 0]) <= 1
    assert 0 <= macro_ece(y, p) <= 1
    assert np.all((sigmoid(np.array([-10.0, 0.0, 10.0])) >= 0) & (sigmoid(np.array([-10.0, 0.0, 10.0])) <= 1))


def test_external_validation_label_contract():
    assert len(LABELS) == 8
    assert LABELS[0] == "Atelectasis"
    assert LABELS[-1] == "Pneumothorax"
