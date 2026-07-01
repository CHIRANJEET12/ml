import numpy as np


def as_matching_arrays(X, y, *, dtype=None):
    X = np.asarray(X, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    if X.size == 0:
        raise ValueError("X and y must not be empty.")

    return X, y


