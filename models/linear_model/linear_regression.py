import numpy as np
from metrics.r2_score import (R2_score, Adjusted_R2_Score)
from models.helper._helper import as_matching_arrays


class LinearRegression:
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights_ = None
        self.bias_ = None
    
    def fit(self, X_train, y_train):

        X_train, y_train = as_matching_arrays(X_train, y_train, dtype=np.float64)

        self.n_samples, self.n_features = X_train.shape

        self.weights_ = np.zeros(self.n_features)
        self.bias_= 0.0

        for i in range(self.epochs):
            y_pred = X_train @ self.weights_ + self.bias_

            error = y_pred - y_train

            dw = (2/self.n_samples) * (X_train.T @ error)
            db = (2/self.n_samples) * np.sum(error)

            self.weights_ -= self.learning_rate * dw
            self.bias_  -= self.learning_rate * db
        
        return self


    def predict(self, X_test):

        X_test = np.asarray(X_test)

        return X_test @ self.weights_ + self.bias_


    def score(self, X_test, y_test, is_r2=True):

        y_pred = self.predict(X_test)
        
        if is_r2:
            r2_score = R2_score()
            return r2_score(y_test, y_pred)
        else:
            adjusted_r2_score = Adjusted_R2_Score()
            return adjusted_r2_score(y_test, y_pred, self.n_samples, self.n_features)
