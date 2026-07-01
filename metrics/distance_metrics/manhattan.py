import numpy as np
from .minkowski import Minkowski
from .validate import Validate

class Manhattan(Minkowski):

    def __init__(self):
        pass

    def __call__(self, X_train, y_train):

        X_train, y_train = Validate.validate(X_train, y_train)

        return np.sum(np.abs(X_train - y_train), axis=1)
        