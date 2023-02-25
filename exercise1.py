import matplotlib_inline
import numpy as np
import pandas as pd
import sklearn.linear_model as skl_lm
import matplotlib.pyplot as plt

pd.set_option('display.width', 200)
pd.set_option('display.max_columns',15)
# To get nicer plots
from IPython.display import set_matplotlib_formats
matplotlib_inline.backend_inline.set_matplotlib_formats('svg') # Output as svg. Else you can try png
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
np.set_printoptions(precision=3);

X = np.array([[1, 2], [1, 3], [1, 4]])
X1 = np.array([2, 3, 4]).reshape(-1,1)
#print(X1)
#print(X)
Y = np.array([-1, 1, 2])
Y1 = np.array([[-1, 0], [1, 2], [2, -1]])



theta_hat = np.linalg.solve((X.T@X), X.T@Y)
#print(theta_hat)


y_hat = theta_hat@np.array([1, 5])
#print(y_hat)


plt.plot(X[:, 1], Y, 'o')
prediction= X@theta_hat
#print(prediction)

plt.plot(X[:, 1], prediction)
#plt.show()


url = 'https://github.com/uu-sml/course-sml-public/raw/master/data/auto.csv'
auto = pd.read_csv(url, na_values='?').dropna()
print(auto.info())
N = auto.shape[0]

np.random.seed(1)
random_index = np.random.choice(N, size=200, replace= False)

Itrain = auto.index.isin(random_index)
train = auto.iloc[Itrain]
test = auto.iloc[~Itrain]

#Linear
model = skl_lm.LinearRegression(fit_intercept= True)

X_train = train.iloc[:, np.arange(1,8)]
Y_train = train[['mpg']]
Y_test = test[['mpg']]
model.fit(X_train, Y_train)
print(model)

#Evaluate on training data
train_predict = model.predict(X_train)

train_RMSE = np.sqrt(np.mean((train_predict-Y_train)**2))
print(f'Train RMSE: {train_RMSE[0]}')

#Evaluate on test data

X_test = test.iloc[:, np.arange(1,8)]
test_predict = model.predict(X_test)
test_RMSE = np.sqrt(np.mean((test_predict-Y_test)**2))
print(f' Test RMSE: {test_RMSE[0]}')

#example of origin variable
sample=auto.origin.sample(30)
print(sample.tolist(),'\n')
Train_dummy_origin = pd.get_dummies(train, columns=['origin'])
print('X after transformation ( origin has been split in three dummy variables): ')
print(Train_dummy_origin.head(30), '\n')

# pick out the input variables

X1_train = Train_dummy_origin[['cylinders',  'displacement',  'horsepower',  'weight',  'acceleration', 'year', 'origin_1',  'origin_2',  'origin_3']]
# also works: X1_train = dummy_origin.drop(['name', 'mpg'], axis=1)

Test_dummy = pd.get_dummies(test, columns=['origin'])
X1_test = Test_dummy.drop(['mpg', 'name'], axis=1)

model1 = skl_lm.LinearRegression()
model1.fit(X1_train, Y_train)

# Evaluate on training data
train_predict1 = model1.predict(X1_train)
train_RMSE1 = np.sqrt(np.mean((train_predict1-Y_train)**2))
print(f'Train RMSE with dummys: {train_RMSE1}')

# Evaluate on test data
test_predict1 = model1.predict(X1_test)
test_RMSE1 = np.sqrt(np.mean((test_predict1-Y_test)**2, axis=0))
print(f'Test RMSE with dummys: {test_RMSE1}')


def computeRMSE(model, X, Y):
    Y_predict = model.predict(X)
    RMSE = np.sqrt(np.mean((Y_predict-Y)**2, axis =0))
    return RMSE

def RMSE_with_drop_col(model, X, Y, X_test, Y_test, drop_col):
    print(f'Results without the variable(s): {drop_col}')
    X = X.drop(drop_col, axis =1)
    model.fit(X, Y)
    train_RMSE = computeRMSE(model, X, Y)
    print(f'Train RMSE: {train_RMSE[0]}')

    X_test = X_test.drop(drop_col, axis = 1)
    test_RMSE = computeRMSE(model, X_test, Y_test)
    print(f'Test RMSE: {test_RMSE[0]}')



# Remove weight

model2 = skl_lm.LinearRegression()

RMSE_with_drop_col(model2, X_train, Y_train, X_test, Y_test, ['weight', 'acceleration'])
# Remove year

model3 = skl_lm.LinearRegression()

RMSE_with_drop_col(model3, X_train, Y_train, X_test, Y_test, ['year'])
# Remove weight

model4 = skl_lm.LinearRegression()

RMSE_with_drop_col(model4, X_train, Y_train, X_test, Y_test, ['weight'])
# Remove weight

model5 = skl_lm.LinearRegression()

RMSE_with_drop_col(model5, X_train, Y_train, X_test, Y_test, ['acceleration'])

model6 = skl_lm.LinearRegression()

RMSE_with_drop_col(model6, X_train, Y_train, X_test, Y_test, ['origin'])

model6 = skl_lm.LinearRegression()

RMSE_with_drop_col(model6, X_train, Y_train, X_test, Y_test, ['cylinders'])

model6 = skl_lm.LinearRegression()

RMSE_with_drop_col(model6, X_train, Y_train, X_test, Y_test, ['displacement'])

model6 = skl_lm.LinearRegression()

RMSE_with_drop_col(model6, X_train, Y_train, X_test, Y_test, ['horsepower'])

model7 = skl_lm.LinearRegression(fit_intercept=True)

RMSE_with_drop_col(model7, X_train, Y_train, X_test, Y_test, ['horsepower'])







