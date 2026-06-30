import numpy as np

from ..utils.losses import (
    MSELoss,
    MAELoss,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    HingeLoss,
)


def separator(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_mse():
    separator("MSE Loss")

    loss = MSELoss()

    y_true = np.array([2, 4, 6])
    y_pred = np.array([3, 5, 7])

    print("y_true     :", y_true)
    print("y_pred     :", y_pred)
    print("Loss       :", loss(y_true, y_pred))
    print("Gradient   :", loss.gradient(y_true, y_pred))


def test_mae():
    separator("MAE Loss")

    loss = MAELoss()

    y_true = np.array([2, 4, 6])
    y_pred = np.array([3, 2, 7])

    print("y_true     :", y_true)
    print("y_pred     :", y_pred)
    print("Loss       :", loss(y_true, y_pred))
    print("Gradient   :", loss.gradient(y_true, y_pred))


def test_binary_cross_entropy():
    separator("Binary Cross Entropy")

    loss = BinaryCrossEntropy()

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.2, 0.8, 0.1])

    print("y_true     :", y_true)
    print("y_pred     :", y_pred)
    print("Loss       :", loss(y_true, y_pred))
    print("Gradient   :", loss.gradient(y_true, y_pred))


def test_categorical_cross_entropy():
    separator("Categorical Cross Entropy")

    loss = CategoricalCrossEntropy()

    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])

    y_pred = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1]
    ])

    print("y_true:")
    print(y_true)

    print("\ny_pred:")
    print(y_pred)

    print("\nLoss:")
    print(loss(y_true, y_pred))

    print("\nGradient:")
    print(loss.gradient(y_true, y_pred))


def test_hinge():
    separator("Hinge Loss")

    loss = HingeLoss()

    y_true = np.array([1, -1, 1])
    y_pred = np.array([2, -0.5, -3])

    print("y_true     :", y_true)
    print("y_pred     :", y_pred)
    print("Loss       :", loss(y_true, y_pred))
    print("Gradient   :", loss.gradient(y_true, y_pred))


def main():
    test_mse()
    test_mae()
    test_binary_cross_entropy()
    test_categorical_cross_entropy()
    test_hinge()


if __name__ == "__main__":
    main()