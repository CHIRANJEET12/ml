import numpy as np
from models.helper._helper import as_matching_arrays
from metrics import (Euclidean, Chebyshev, Manhattan, Minkowski, Accuracy)

class KNN:

    def __init__(self):
        self.X = None
        self.y = None
        self.k = None

    def _compute_distance(self,X_test,val):

        if val == 1:
            return Euclidean()(self.X, X_test)
        elif val == 2:
            return Manhattan()(self.X, X_test)
        elif val == 3:
            return Minkowski()(self.X, X_test)
        else:
            return Chebyshev()(self.X, X_test)

        

    def fit(self, X_train, y_train, k=3):

        X_train, y_train = as_matching_arrays(X_train, y_train)

        self.X = X_train

        self.y = y_train

        self.k = k

        self.unique_classes = np.unique(self.y)



    def predict(self, X_test, val=1):

        X_test = np.asarray(X_test)
        
        distance = self._compute_distance(X_test,val)

        k_indices = np.argsort(distance)[:self.k]

        k_nearest_labels = self.y[k_indices]

        most_common_label = None
        max_count = -1

        for i in self.unique_classes:
            
            count = np.sum(k_nearest_labels == i)
            if count > max_count:
                max_count = count
                most_common_label = i


        return most_common_label


    def predict_proba(self, X_test, val=1):
        distance = self._compute_distance(X_test,val)

        k_indices = np.argsort(distance)[:self.k]

        k_nearest_labels = self.y[k_indices]

        prob_arr = []


        for i in self.unique_classes:

            count = np.sum(k_nearest_labels == i)
            prob_arr.append(count / self.k)
        
        return prob_arr


    def score(self, X_test, y_test,val=1):

        y_pred = self.predict(X_test,val)

        return Accuracy()(y_test, y_pred)
        