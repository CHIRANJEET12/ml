import numpy as np

from .mse import Mean_Squared_Error

class RMSE:
    def __call__(self, y_test, y_pred):
        mse = Mean_Squared_Error()
        return np.sqrt(mse(y_test, y_pred))

        
Root_Mean_Squared_Error = RMSE