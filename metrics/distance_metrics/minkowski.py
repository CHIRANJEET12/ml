from abc import ABC, abstractmethod
import numpy as np

class Minkowski(ABC):
    
    @abstractmethod
    def __call__(self, X_train, y_train):
        pass

