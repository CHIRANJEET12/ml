import numpy as np
from .minkowski import Minkowski
from .validate import Validate

class Euclidean(Minkowski):

    def __init__(self):
        pass

    def __call__(self, X_train, X_test):

        X_train, X_test = Validate.validate(X_train, X_test)

        return np.sqrt(np.sum((X_train - X_test) ** 2, axis=1))
        