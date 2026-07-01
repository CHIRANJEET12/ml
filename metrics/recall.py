import numpy as np

from .confusion_matrix import Confusion_Matrix
from .precision import _average_scores, _safe_divide


class Recall:
    def __call__(self, y_test, y_pred, positive_label=1, average="binary"):
        labels = None if average != "binary" else [0, positive_label]
        matrix = Confusion_Matrix()(y_test, y_pred, labels=labels)
        labels = np.array([0, positive_label]) if average == "binary" else np.unique(np.concatenate([np.ravel(y_test), np.ravel(y_pred)]))
        per_class = _safe_divide(np.diag(matrix), matrix.sum(axis=1))
        return _average_scores(per_class, matrix, labels, positive_label, average)

