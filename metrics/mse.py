import numpy as np

from ._utils import as_matching_arrays


class MSE:
    def __call__(self, y_test, y_pred):
        y_test, y_pred = as_matching_arrays(y_test, y_pred, dtype=np.float64)
        return np.mean((y_test - y_pred) ** 2)


Mean_Squared_Error = MSE

