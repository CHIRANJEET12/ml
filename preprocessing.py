import numpy as np


class Preprocessing:

    def __init__(self):
        pass

    def train_test_split(self,X,y,test_size=0.2,shuffle=True,random_state=42,train_size=None):

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
