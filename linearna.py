# linearna

import pandas as pd
import numpy as np

df = pd.read_csv('data.csv', header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
df.insert(0, 'one', 1)
df = df.sample(frac=1)

predictors = 5
boundary_index = round(df.shape[0] * 0.8)

X = df.iloc[:, 0:6].to_numpy()
Y = df['y'].to_numpy()

J_arr = []

for i in range(0, predictors):
    if i > 1:
        for j in range(1, 6):  # first col is all ones
            X = np.c_[X, (X[:, j] ** i)]

    X_train = X[0:boundary_index]
    Y_train = Y[0:boundary_index]

    X_test = X[boundary_index:df.shape[0] - 1]
    Y_test = Y[boundary_index:df.shape[0] - 1]

    theta = np.linalg.inv((X_train.T.dot(X_train))).dot(X_train.T).dot(Y_train)

    y_guess = X_test.dot(theta)
    print(i)
    print(y_guess)
    print(Y_test)
    # y = np.linspace(Y_test)
    J = (Y - X.dot(theta)).T.dot(Y - X.dot(theta))
    print(J)


# prvi model obican, drugi: x1^2, x2^2..., treci: x1^3, x2^3
