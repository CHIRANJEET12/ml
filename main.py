import numpy as np
from preprocessing import Preprocessing
from sklearn.model_selection import train_test_split

X = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40],
    [5, 50]
])

y = np.array([0, 1, 0, 1, 0])

p = Preprocessing()

X_train, X_test, y_train, y_test = p.train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

print("X_train")
print(X_train)

print("\nX_test")
print(X_test)

print("\ny_train")
print(y_train)

print("\ny_test")
print(y_test)






#sklear
# X_train1, X_test1, y_train1, y_test1 = train_test_split(
#     X,
#     y,
#     test_size=0.4,
#     random_state=42
# )

# print("X_train")
# print(X_train1)

# print("\nX_test")
# print(X_test1)

# print("\ny_train")
# print(y_train1)

# print("\ny_test")
# print(y_test1)
