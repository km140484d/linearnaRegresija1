# linearna

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv', header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
df.insert(0, 'one', 1)
# df = df.sample(frac=1)

# polynomial
predictors = 5
boundary_index = round(df.shape[0] * 0.8)

X = df.iloc[:, 0:6].to_numpy()

Y = df['y'].to_numpy()

J_arr = []
theta_arr = []
pred_arr = []

for i in range(1, 6):
    print('PREDICTORS ', predictors)
    if 1 < i < 5:
        for j in range(1, 6):  # first col is all ones
            for k in range(j, 6):
                if i == 2:
                    X = np.c_[X, (X[:, j] * X[:, k])]
                    predictors = predictors + 1
                    print('x', j, ' x', k)
                else:
                    for l in range(k, 6):
                        if i == 3:
                            predictors = predictors + 1
                            print('x', j, ' x', k, 'x', l)
                            X = np.c_[X, (X[:, j] * X[:, k] * X[:, l])]
                        else:
                            for m in range(l, 6):
                                if i == 4:
                                    predictors = predictors + 1
                                    print('x', j, ' x', k, 'x', l, 'x', m)
                                    X = np.c_[X, (X[:, j] * X[:, k] * X[:, l] * X[:, m])]
    else:
        if i == 5:
            X = np.c_[X, (X[:, 1] * X[:, 2] * X[:, 3] * X[:, 4] * X[:, 5])]
            predictors = predictors + 1
            print('x1 x2 x3 x4 x5')

    pred_arr.append(predictors)

    X_train, Y_train = X[0:boundary_index], Y[0:boundary_index]
    X_test, Y_test = X[boundary_index:df.shape[0]], Y[boundary_index:df.shape[0]]

    theta = np.linalg.inv((X_train.T.dot(X_train))).dot(X_train.T).dot(Y_train)
    theta_arr.append(theta)

    y_guess = X_test.dot(theta)
    J = (Y - X.dot(theta)).T.dot(Y - X.dot(theta)) / Y.shape[0]
    J_arr.append(J)

J_min_index = J_arr.index(min(J_arr))
theta_opt = theta_arr[J_min_index]
print(pred_arr[J_min_index])
print(np.min(J_arr))

# plt.plot(pred_arr, J_arr, label='polynomial')
# plt.legend()
# plt.show()

# grebena regresija
X = X[:, 0:pred_arr[J_min_index]]
X_train = X_train[:, 0:pred_arr[J_min_index]]
X_test = X_test[:, 0:pred_arr[J_min_index]]

lam = np.linspace(0, 0.5, 101)
J_ridge_arr = []
theta_ridge_arr = []
for i in lam:
    theta = np.linalg.inv((X_train.T.dot(X_train) + i * np.eye(pred_arr[J_min_index]))).dot(X_train.T).dot(Y_train)
    theta_ridge_arr.append(theta)
    J = (Y - X.dot(theta)).T.dot(Y - X.dot(theta)) / Y.shape[0]
    J_ridge_arr.append(J)

J_min_index = J_arr.index(min(J_arr))
print(lam[J_min_index])
theta_opt = theta_arr[J_min_index]
lam_opt = lam[J_min_index]

# plt.plot(lam, J_ridge_arr, label='ridge')
# plt.legend()
# plt.show()

tau = np.linspace(0.1, 5, 5)
xa = np.repeat([X_test[0]], repeats=len(X_train), axis=0)
for t in tau:
    for x in X_test:
        Xi = np.repeat([x], repeats=len(X_train), axis=0)
        exp = []
        for i in range(len(X_train)):
            exp.append(math.e**(-np.linalg.norm(Xi[i] - X_train[i])**2/(2*t**2)))
        W = np.diag(exp)
        # theta = np.linalg.inv(X_test.T.dot(W).dot(X_test)).dot(X_test).T.dot(W).dot(Y_test)
        # print('theta', theta)
        # print('W', W)
        exit(0)


# prvi model obican, drugi: x1^2, x2^2..., treci: x1^3, x2^3
