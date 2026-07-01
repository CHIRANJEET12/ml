import numpy as np

from ._utils import as_binary_arrays


class Binary_Log_Loss:
    def __call__(self, y_test, y_prob, eps=1e-15):
        y_test, y_prob = as_binary_arrays(y_test, y_prob)
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(y_test * np.log(y_prob) + (1 - y_test) * np.log(1 - y_prob))


Log_Loss = Binary_Log_Loss

