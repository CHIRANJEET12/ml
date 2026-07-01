import numpy as np


def as_matching_arrays(y_test, y_pred, *, dtype=None):
    y_test = np.asarray(y_test, dtype=dtype)
    y_pred = np.asarray(y_pred, dtype=dtype)

    if y_test.shape != y_pred.shape:
        raise ValueError("y_test and y_pred must have the same shape.")

    if y_test.size == 0:
        raise ValueError("y_test and y_pred must not be empty.")

    return y_test, y_pred


def as_binary_arrays(y_test, y_score=None):
    y_test = np.asarray(y_test).ravel()

    if y_test.size == 0:
        raise ValueError("y_test must not be empty.")

    unique = np.unique(y_test)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("y_test must contain only binary labels 0 and 1.")

    if y_score is None:
        return y_test.astype(int)

    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_test.shape != y_score.shape:
        raise ValueError("y_test and y_score must have the same shape.")

    return y_test.astype(int), y_score


def sorted_labels(y_test, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate([y_test.ravel(), y_pred.ravel()]))
    else:
        labels = np.asarray(labels)

    if labels.size == 0:
        raise ValueError("labels must not be empty.")

    return labels

