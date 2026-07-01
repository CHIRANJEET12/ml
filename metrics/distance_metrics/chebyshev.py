import numpy as np
from .minkowski import Minkowski
from .validate import Validate


class Chebyshev(Minkowski):

    def __init__(self):
        pass

    def __call__(self, X_train, X_test):

        X_train, X_test = Validate.validate(X_train, X_test)

        return np.max(np.abs(X_train - X_test), axis=1)
        