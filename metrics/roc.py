import numpy as np

from ._utils import as_binary_arrays


class ROC:
    def __call__(self, y_test, y_score):
        y_test, y_score = as_binary_arrays(y_test, y_score)
        thresholds = np.r_[np.inf, np.unique(y_score)[::-1]]
        positives = np.sum(y_test == 1)
        negatives = np.sum(y_test == 0)

        if positives == 0 or negatives == 0:
            raise ValueError("ROC is undefined when y_test has only one class.")

        tpr = []
        fpr = []

        for threshold in thresholds:
            y_pred = y_score >= threshold
            true_positive = np.sum((y_pred == 1) & (y_test == 1))
            false_positive = np.sum((y_pred == 1) & (y_test == 0))
            tpr.append(true_positive / positives)
            fpr.append(false_positive / negatives)

        return np.asarray(fpr), np.asarray(tpr), thresholds

