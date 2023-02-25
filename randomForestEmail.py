import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn.tree
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import graphviz
import tempfile
from matplotlib.image import imread

# Nice options
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 15)  # see up to 15 columns
np.set_printoptions(suppress=True)  # shows 0.07 instead of 7*e^-2

url = 'https://uu-sml.github.io/course-sml-public/data/email.csv'
email = pd.read_csv(url)

np.random.seed(1)
N = email.shape[0]
n = round(N * 0.75)
trainIndex = np.random.choice(N, n, replace=False)
train_bool = email.index.isin(trainIndex)

train = email.iloc[train_bool]
test = email.iloc[~train_bool]
x_train = train.drop('Class', axis=1)
y_train = train['Class']
x_test = test.drop('Class', axis=1)
y_test = test['Class']

# Decision tree
model = tree.DecisionTreeClassifier(max_leaf_nodes=5)
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
accuracy = np.mean(y_prediction == y_test)
conf_matrix = pd.crosstab(y_prediction, y_test)
print(f'Decision tree accuracy = {accuracy}')
print(f'confusion matrix: \n{conf_matrix}')
print('\n')

# Bagging classifier
model = BaggingClassifier()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
accuracy = np.mean(y_prediction == y_test)
conf_matrix = pd.crosstab(y_prediction, y_test)
print(f'Bagging classifier accuracy = {accuracy}')
print(f'confusion matrix: \n{conf_matrix}')
print('\n')

# Random forest classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
accuracy = np.mean(y_prediction == y_test)
conf_matrix = pd.crosstab(y_prediction, y_test)
print(f'Random forest accuracy = {accuracy}')
print(f'confusion matrix: \n{conf_matrix}')
print('\n')

# Bootstrap
"""
np.random.seed(0)
n = 100
y = np.random.normal(4, 1, n)
theta_zero = np.mean(y)
print(theta_zero)
theta = []
theta1 = []
for i in range(1000):
    theta.append(np.mean(np.random.normal(4, 1, n)))
    indices = np.random.choice(np.arange(100), n, replace=True)
    y_new = y[indices]
    theta1.append(np.mean(y_new))


plt.figure(1)
plt.hist(theta, bins=14)

plt.figure(2)
plt.hist(theta1, bins=14)

plt.show()
"""