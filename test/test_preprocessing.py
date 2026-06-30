import numpy as np
from ..preprocessing import Preprocessing, StandardScaler
# from sklearn.model_selection import train_test_split

X = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40],
    [5, 50]
])

y = np.array([0, 1, 0, 1, 0])

p = Preprocessing()
scaler = StandardScaler()


# X_train, X_test, y_train, y_test = p.train_test_split(
#     X,
#     y,
#     test_size=0.4,
#     random_state=42
# )

# print("X_train")
# print(X_train)

# print("\nX_test")
# print(X_test)

# print("\ny_train")
# print(y_train)

# print("\ny_test")
# print(y_test)


# scaler = StandardScaler()

# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


# print("X_train")
# print(X_train)

# print("\nX_test")
# print(X_test)

# X_train = np.round(scaler.inverse_transform(X_train)).astype(int)
# X_test = np.round(scaler.inverse_transform(X_test)).astype(int)


# print("X_train")
# print(X_train)

# print("\nX_test")
# print(X_test)

