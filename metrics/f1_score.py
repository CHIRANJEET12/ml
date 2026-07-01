import numpy as np

from .precision import Precision
from .recall import Recall


class F1:
    def __call__(self, y_test, y_pred, positive_label=1, average="binary"):
        precision = Precision()(y_test, y_pred, positive_label, average)
        recall = Recall()(y_test, y_pred, positive_label, average)

        precision = np.asarray(precision, dtype=np.float64)
        recall = np.asarray(recall, dtype=np.float64)
        denominator = precision + recall
        scores = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator != 0,
        )

        return float(scores) if scores.ndim == 0 else scores


F1_Score = F1

