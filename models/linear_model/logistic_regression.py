import numpy as np
from models.helper._helper import as_matching_arrays
from utils.losses import BinaryCrossEntropy
from metrics.accuracy import Accuracy


class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights_ = None
        self.bias_ = None
        self.loss = None
    
    def _sigmoid(self, z):

        z = np.clip(z,-500,500)

        return 1 / (1 + np.exp(-z))
        
    def fit(self, X_train, y_train):

        if not np.all(np.isin(y_train,[0,1])):
            raise ValueError(
                "y_train contains only 0, 1 Otherwise BCE is invalid."
            )

        X_train, y_train = as_matching_arrays(X_train, y_train, dtype=np.float64)

        self.n_samples, self.n_features = X_train.shape

        bce = BinaryCrossEntropy()

        self.weights_ = np.zeros(self.n_features)
        self.bias_ = 0

        print("Losses : ")


        for i in range(self.epochs):
            if i % 100 == 0:
                print(self.loss)

            z = X_train @ self.weights_ + self.bias_

            y_pred = self._sigmoid(z)

            error = y_pred - y_train

            dw = (1/self.n_samples) * (X_train.T @ error) 
            db = (1/self.n_samples) * np.sum(error)

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            self.loss = bce(y_train, y_pred)
        
        return self
    

    def predict_proba(self, X_test):
        X_test = np.asarray(X_test)

        z = X_test @ self.weights_ + self.bias_

        prob_1 = self._sigmoid(z)

        prob_0 = 1 - prob_1

        probabilities = np.column_stack((prob_0, prob_1))

        return probabilities

    def predict(self, X_test, threshold=0.5):
        X_test = np.asarray(X_test)

        prob = self.predict_proba(X_test)[:,1]

        return (prob >= threshold).astype(int)


    def score(self, X_test, y_test):

        y_pred = self.predict(X_test)

        y_pred = np.array(y_pred)
        y_test = np.array(y_test)

        return Accuracy()(y_test, y_pred)









