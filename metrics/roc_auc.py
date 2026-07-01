import numpy as np

from .roc import ROC


class ROC_AUC:
    def __call__(self, y_test, y_score):
        fpr, tpr, _ = ROC()(y_test, y_score)
        order = np.argsort(fpr)
        return np.trapezoid(tpr[order], fpr[order])


ROCAUC = ROC_AUC

