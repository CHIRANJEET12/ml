import numpy as np

from ._utils import as_matching_arrays


class MAPE:
    def __call__(self, y_test, y_pred):
        y_test, y_pred = as_matching_arrays(y_test, y_pred, dtype=np.float64)

        if np.any(y_test == 0):
            raise ValueError("MAPE is undefined when y_test contains zero.")

        return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


Mean_Absolute_Percentage_Error = MAPE

