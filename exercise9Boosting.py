import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


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

model = AdaBoostClassifier()
model.fit(x_train,y_train)
y_prediction = model.predict(x_test)
accuracy = np.mean(y_prediction == y_test)
conf_matrix = pd.crosstab(y_prediction, y_test)
print(f'AdaBoost accuracy = {accuracy}')
print(f'confusion matrix: \n{conf_matrix}')
print('\n')
