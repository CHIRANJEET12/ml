import numpy as np
from abc import ABC, abstractmethod

class Score(ABC):

    
    @abstractmethod
    def __call__(self, y_test, y_pred, n_samples, n_features):
        """Implements Scores"""
        pass


class R2_score(Score):


    
    def __call__(self, y_test, y_pred, n_samples=None, n_features=None):
        y_test = np.asarray(y_test, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if y_test.shape != y_pred.shape:
            raise ValueError(
                "y_true and y_pred must have the same shape."
            )
        
        y_avg = np.average(y_test)
        rss = np.sum((y_pred - y_test)**2)    
        tss = np.sum((y_test-y_avg)**2)

        if tss == 0:
            return 0.0

        r2 = 1 - (rss/tss)

        return r2

class Adjusted_R2_Score(Score):


    def __call__(self, y_test, y_pred, n_samples, n_features):

        if n_samples <= n_features + 1:
            raise ValueError(
                "Adjusted R² is undefined when "
                "n_samples <= n_features + 1."
            )
        
        r2_score = R2_score()
        
        adjusted_r2 = 1 - ((1 - r2_score(y_test, y_pred)) * (n_samples - 1) / (n_samples - n_features - 1))

        return adjusted_r2