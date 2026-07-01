import numpy as np

from ._utils import as_matching_arrays, sorted_labels


class Confusion_Matrix:
    def __call__(self, y_test, y_pred, labels=None):
        y_test, y_pred = as_matching_arrays(y_test, y_pred)
        labels = sorted_labels(y_test, y_pred, labels)
        label_to_index = {label: index for index, label in enumerate(labels)}
        matrix = np.zeros((labels.size, labels.size), dtype=int)

        for actual, predicted in zip(y_test.ravel(), y_pred.ravel()):
            if actual not in label_to_index or predicted not in label_to_index:
                raise ValueError("labels must include every value in y_test and y_pred.")

            matrix[label_to_index[actual], label_to_index[predicted]] += 1

        return matrix


ConfusionMatrix = Confusion_Matrix

