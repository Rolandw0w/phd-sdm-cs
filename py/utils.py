import numpy as np

from sklearn.metrics import confusion_matrix


def calculate_l1(x: np.ndarray, y: np.ndarray):
    return np.abs(x-y).sum()


def perf_measure(y_true: np.array, y_pred: np.array):
    assert y_true.shape == y_pred.shape

    fn = 0
    fp = 0
    tn = 0
    tp = 0
    for i in range(y_true.shape[0]):
        y_true_i = y_true[i]
        y_pred_i = y_pred[i]

        if y_true_i == 0 and y_pred_i == 0:
            tn += 1
        if y_true_i == 0 and y_pred_i == 1:
            fp += 1
        if y_true_i == 1 and y_pred_i == 0:
            fn += 1
        if y_true_i == 1 and y_pred_i == 1:
            tp += 1

    return fn, fp, tn, tp


def perf_measure_sklearn(y_true: np.array, y_pred: np.array):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return fn, fp, tn, tp
