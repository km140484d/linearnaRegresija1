# linearna

import pandas as pd
import numpy as np

df = pd.read_csv('data.csv', header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
df.insert(0, 'one', 1)
# df = df.sample(frac=1)

predictors = 9
boundary_index = round(df.shape[0] * 0.9)

X = df.iloc[:, 0:6].to_numpy()
Y = df['y'].to_numpy()

ten_per = round(df.shape[0] * 0.1)

for i in range(1, predictors):
    if i > 1:
        for j in range(1, 6):  # first col is all ones
            a = X[0:5, j] ** i
            X = np.c_[X, (X[:, j] ** i)]
    # print(X)
    print(i)
    J_array = []
    for k in range(0, 4):
        X_test = X[ten_per*k:ten_per*(k+1)]
        Y_test = Y[ten_per*k:ten_per*(k+1)]
        # print(X_test)

        X_train = np.append(X[0:ten_per*k], X[ten_per*(k+1):])
        Y_train = np.append(Y[0:ten_per*k], Y[ten_per*(k+1):])

        print(X_test)
        print()
        # print(Y_train)

    #     theta = np.linalg.inv((X_train.T.dot(X_train))).dot(X_train.T).dot(Y_train)
    #     # print(theta)
    #
    #     y_guess = X_test.dot(theta)
    #     # print(y_guess)
    #     # print(Y_test)
    #     # y = np.linspace(Y_test)
    #     J = (Y - X.dot(theta)).T.dot(Y - X.dot(theta))
    #     J_array.append(J)
    # print(J / len(J_array))
    # print()
    # print()

# X = df[['one', 'x1', 'x2', 'x3', 'x4', 'x5']].to_numpy()

# prvi model obican, drugi: x1^2, x2^2..., treci: x1^3, x2^3
