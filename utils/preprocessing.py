import numpy as np
from abc import ABC, abstractmethod

class Process(ABC):

    @abstractmethod
    def __call__(self, X,y,test_size=0.2,shuffle=True,random_state=42,train_size=None):
        pass

class Preprocessing(Process):

    def __init__(self):
        pass

    def __call__(self, X, y, test_size=0.2, shuffle=True, random_state=42, train_size=None):
        """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array-like
        Feature matrix of shape (n_samples, n_features)

    y : array-like
        Target vector of shape (n_samples,)

    test_size : float
        Fraction of samples used for testing.

    shuffle : bool
        Whether to shuffle before splitting.

    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
        n_samples = len(X)
        X = np.asarray(X)
        y = np.asarray(y)

        if test_size > 1 or test_size < 0:
            raise ValueError(
                "Test size should be in the range 0 < test_size < 1"
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )
        

        indice = np.arange(len(X))


        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(indice)

        if train_size is not None:
            if train_size <= 0 or train_size >= 1:
                raise ValueError(
                    "train_size must be between 0 and 1."
                )
            n_train = int(n_samples * train_size)
            n_test = int(n_samples - n_train)
        else:
            n_test = int(n_samples * test_size)
            n_train = int(n_samples - n_test)



        train_idx = indice[:n_train]
        test_idx = indice[n_train:]



        return (
            X[train_idx],
            X[test_idx],
            y[train_idx],
            y[test_idx]
        )

class BaseEstimator:
    def __init__(self):
        self.is_fitted_ = False
    
    def get_is_fitted(self):
        if not self.is_fitted_:
            raise ValueError(
                "Estimator has not been fitted yet."
            )


class StandardScaler(BaseEstimator):

    def __init__(self):
        super().__init__()
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None

    def fit(self, X):

        self.is_fitted_ = True

        X = np.asarray(X)

        sample = len(X)

        self.n_features_ = X.shape[1]

        self.mean_ = np.mean(X, axis=0)

        self.std_ = np.std(X, axis=0)

    def transform(self, X):

        self.get_is_fitted()
        
        X = np.asarray(X)

        mean = self.mean_
        std = self.std_

        up = X - mean

        z = up / std

        return z
    
    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)
    
    def inverse_transform(self, X_scaled):

        self.get_is_fitted()
        
        X_scaled = np.asarray(X_scaled)
        
        X = X_scaled*self.std_ + self.mean_

        return X

