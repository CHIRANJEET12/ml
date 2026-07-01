import numpy as np

from ._utils import as_matching_arrays


class Accuracy:
    def __call__(self, y_test, y_pred):
        y_test, y_pred = as_matching_arrays(y_test, y_pred)
        return np.mean(y_test == y_pred)

