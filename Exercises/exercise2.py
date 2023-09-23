import matplotlib_inline
import numpy as np
import pandas as pd
import sklearn.linear_model as skl_lm
import matplotlib.pyplot as plt

np.random.seed(1)
x_train = np.random.uniform(0, 10, 100)

y_train = .4 - .6*x_train + 3.*np.sin(x_train - 1.2) + np.random.normal(0, 0.1, 100)

X_train = np.column_stack([np.ones(100), x_train, np.cos(x_train), np.sin(x_train)])

y_train = np.array(y_train).reshape((-1,1))
theta_hat = np.linalg.solve(X_train.T@X_train,X_train.T@y_train)
print(f'Theta hat: {theta_hat}')

#Prediction

x_test = np.linspace(0,10,100)
X_test = np.column_stack([np.ones(100), x_test, np.cos(x_test), np.sin(x_test)])
y_test_hat = X_test@theta_hat


plt.figure(1)
plt.plot(x_train,y_train, 'o', label='training data')
plt.plot(x_test, y_test_hat, 'o', label='test data')
plt.legend()
plt.title('Training data')
plt.xlabel('input x')
plt.ylabel('output y')

#plt.show()

model = skl_lm.LinearRegression()
model.fit(X_train,y_train)