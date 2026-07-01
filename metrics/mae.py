import numpy as np

from ._utils import as_matching_arrays


class MAE:
    def __call__(self, y_test, y_pred):
        y_test, y_pred = as_matching_arrays(y_test, y_pred, dtype=np.float64)
        return np.mean(np.abs(y_test - y_pred))


Mean_Absolute_Error = MAE

