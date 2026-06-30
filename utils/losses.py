from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """Abstract base class for all loss functions."""

    @abstractmethod
    def __call__(self, y_true, y_pred):
        """Return scalar loss."""
        pass

    @abstractmethod
    def gradient(self, y_true, y_pred):
        """Return gradient w.r.t. predictions."""
        pass

    def _validate_inputs(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )

        return y_true, y_pred
    
class MSELoss(Loss):

    def __call__(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        return np.mean((y_true - y_pred) ** 2)
    
    def gradient(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        return -(2 / y_true.size) * (y_true - y_pred)
    
class MAELoss(Loss):

    def __call__(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        return np.mean(np.abs(y_true - y_pred))    
    
    def gradient(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        return -np.sign(y_true - y_pred) / y_true.size
    

class BinaryCrossEntropy(Loss):

    def __call__(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        eps = 1e-15
        y_pred = np.clip(y_pred,eps,1-eps)

        return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    def gradient(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        eps = 1e-15
        y_pred = np.clip(y_pred,eps,1-eps)

        return -(y_true - y_pred) / (y_pred*(1-y_pred)*y_true.size)
    

class CategoricalCrossEntropy(Loss):

    def __call__(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        eps = 1e-15
        y_pred = np.clip(y_pred,eps,1-eps)

        return -np.mean(np.sum(y_true * np.log(y_pred)))
    def gradient(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        eps = 1e-15
        y_pred = np.clip(y_pred,eps,1-eps)

        return -(y_true/y_pred) / y_true.size
    
class HingeLoss(Loss):

    def __call__(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        return np.mean(np.maximum(0, 1-y_pred * y_true))
    
    def gradient(self, y_true, y_pred):
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        grad = np.zeros_like(y_pred)

        mask = (1-y_true*y_pred) > 0

        grad[mask] = -y_true[mask]

        return grad/y_true.size