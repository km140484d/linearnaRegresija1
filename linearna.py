# linearna

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


df = pd.read_csv('data.csv', header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
df.insert(0, 'one', 1)
df = df.sample(frac=1)

boundary_index = round(df.shape[0] * 0.8)       # uzeto je da je 80% skup za treniranje, a 20% skup za testiranje
X, Y = df.iloc[:, 0:6].to_numpy(), df['y'].to_numpy()

# polinomijalna regresija
predictors = 5                                  # inicijalan broj prediktora 5 + 1 (kolona sa 1)
J_arr, theta_arr, predictors_arr = [], [], []

for i in range(1, 6):
    if 1 < i < 5:
        for j in range(1, 6):  # first col is all ones
            for k in range(j, 6):
                if i == 2:
                    X = np.c_[X, (X[:, j] * X[:, k])]
                    predictors = predictors + 1
                else:
                    for l in range(k, 6):
                        if i == 3:
                            X = np.c_[X, (X[:, j] * X[:, k] * X[:, l])]
                            predictors = predictors + 1
                        else:
                            for m in range(l, 6):
                                if i == 4:
                                    X = np.c_[X, (X[:, j] * X[:, k] * X[:, l] * X[:, m])]
                                    predictors = predictors + 1
    predictors_arr.append(predictors)
    X_train, Y_train = X[0:boundary_index], Y[0:boundary_index]
    X_test, Y_test = X[boundary_index:df.shape[0]], Y[boundary_index:df.shape[0]]
    theta = np.linalg.inv((X_train.T.dot(X_train))).dot(X_train.T).dot(Y_train)
    theta_arr.append(theta)
    J_arr.append((Y - X.dot(theta)).T.dot(Y - X.dot(theta)) / Y.shape[0])
J_min_index = J_arr.index(min(J_arr))
print('Polinomijalna, min(J):', J_arr[J_min_index])
print('predictor_number_opt:', predictors_arr[J_min_index])
print(predictors_arr[0:J_min_index + 10])
plt.plot(predictors_arr, J_arr, label='polynomial')
plt.xlabel('n')
plt.ylabel('J')
plt.legend()
plt.show()

# grebena regresija
lam = np.linspace(0, 10, 51)
J_ridge_arr, theta_ridge_arr = [], []
for l in lam:
    theta = np.linalg.inv(X_train.T.dot(X_train) + l * np.eye(len(X[0]))).dot(X_train.T).dot(Y_train)
    theta_ridge_arr.append(theta)
    J_ridge_arr.append((Y - X.dot(theta)).T.dot(Y - X.dot(theta)) / Y.shape[0])

J_min_index = J_ridge_arr.index(min(J_ridge_arr))
theta_opt = theta_arr[J_min_index]
print('Grebena, min(J):', J_ridge_arr[J_min_index])
print('lambda_opt:', lam[J_min_index])
plt.plot(lam, J_ridge_arr, label='ridge')
plt.xlabel('λ')
plt.ylabel('J')
plt.legend()
plt.show()

# podesavanje prediktora, na model koji daje najbolje rezultate u polinomijalnoj regresiji
X = X[:, 0:predictors_arr[J_min_index]]
X_train, X_test = X_train[:, 0:predictors_arr[J_min_index]], X_test[:, 0:predictors_arr[J_min_index]]

# lokalno ponderisana
tau = np.linspace(0.1, 20, 30)
J_pond_arr = []
for t in tau:
    J = 0
    for xi in range(len(X_test)):
        exp = []
        for i in range(len(X_train)):
            exp.append(math.e**(-np.linalg.norm(X_test[xi] - X_train[i])**2/(2*t**2)))
        W = np.diag(exp)
        theta = np.linalg.inv(X_train.T.dot(W).dot(X_train)).dot(X_train.T).dot(W).dot(Y_train)
        J = J + (Y[xi]-X[xi].dot(theta))**2
    J_pond_arr.append(J/Y.shape[0])

J_min_index = J_pond_arr.index(min(J_pond_arr))
print('Lokalno ponderisana, min(J):', J_pond_arr[J_min_index])
print('tau_opt:', round(tau[J_min_index], 5))
plt.plot(tau, J_pond_arr, label='local segmented')
plt.xlabel('τ')
plt.ylabel('J')
plt.legend()
plt.show()
