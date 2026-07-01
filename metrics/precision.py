import numpy as np

from .confusion_matrix import Confusion_Matrix


class Precision:
    def __call__(self, y_test, y_pred, positive_label=1, average="binary"):
        labels, matrix = _labels_and_matrix(y_test, y_pred, positive_label, average)
        per_class = _safe_divide(np.diag(matrix), matrix.sum(axis=0))
        return _average_scores(per_class, matrix, labels, positive_label, average)


def _labels_and_matrix(y_test, y_pred, positive_label, average):
    labels = None if average != "binary" else [0, positive_label]
    matrix = Confusion_Matrix()(y_test, y_pred, labels=labels)
    labels = np.array([0, positive_label]) if average == "binary" else np.unique(np.concatenate([np.ravel(y_test), np.ravel(y_pred)]))
    return labels, matrix


def _safe_divide(numerator, denominator):
    out = np.zeros_like(numerator, dtype=np.float64)
    return np.divide(numerator, denominator, out=out, where=denominator != 0)


def _average_scores(scores, matrix, labels, positive_label, average):
    if average == "binary":
        positive_index = int(np.where(labels == positive_label)[0][0])
        return scores[positive_index]

    if average == "macro":
        return np.mean(scores)

    if average == "weighted":
        support = matrix.sum(axis=1)
        return np.average(scores, weights=support) if support.sum() else 0.0

    if average == "micro":
        true_positive = np.diag(matrix).sum()
        predicted_positive = matrix.sum(axis=0).sum()
        return true_positive / predicted_positive if predicted_positive else 0.0

    if average is None:
        return scores

    raise ValueError("average must be 'binary', 'macro', 'micro', 'weighted', or None.")

