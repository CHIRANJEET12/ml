import numpy as np

class Validate:

    @staticmethod
    def validate(X_train, X_test, dtype=None):
        X_train = np.asarray(X_train, dtype=dtype)
        X_test = np.asarray(X_test, dtype=dtype)

        if X_test.size == 0:
            raise ValueError("X_test must not be empty.")
        
        return X_train, X_test