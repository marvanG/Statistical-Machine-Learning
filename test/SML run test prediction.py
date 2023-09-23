import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import sklearn.tree
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
from sklearn import preprocessing
import graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib.image import imread

# Nice options
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)  # see up to 15 columns
pd.set_option('display.max_rows', 3000)  # see up to 15 columns
np.set_printoptions(suppress=True)  # shows 0.07 instead of 7*e^-2

# Data preparation
df = pd.read_csv(r'C:\Users\mariu\PycharmProjects\machineLearning\train.csv')  # read in data in a panda dataframe
test_df = pd.read_csv(r'C:\Users\mariu\PycharmProjects\machineLearning\test.csv')  # read in data in a panda dataframe

# normalizing techniques
# Drop classifier Lead
classifier = df['Lead']
df_minus_classifier = df.drop('Lead', axis=1)
# RobustScaler:
robust_scaler = preprocessing.RobustScaler()
df_robust = pd.DataFrame(robust_scaler.fit_transform(df_minus_classifier), columns=df_minus_classifier.columns)

test_robust = pd.DataFrame(robust_scaler.fit_transform(test_df), columns=test_df.columns)

# X train
x_train = df_robust.drop(['Number words male', 'Gross', 'Year'], axis=1)

x_test = test_robust.drop(['Number words male', 'Gross', 'Year'], axis=1)
print(x_test.info())


# Quadratic discriminant analysis
model = skl_da.QuadraticDiscriminantAnalysis()

model.fit(x_train, classifier)
# prediction, probability
prediction_probability = model.predict_proba(x_train)

prediction = np.where(prediction_probability[:, 0] > 0.5, 'Female', 'Male')
# Accuracy and confusion matrix
accuracy = np.mean(prediction == classifier)
conf_matrix = pd.crosstab(prediction, classifier)
print(f'Accuracy = {accuracy}')
print(f'confusion matrix:\n{conf_matrix}')

test_prediction_probability = model.predict_proba(x_test)
test_prediction = np.where(test_prediction_probability[:, 0] > 0.5, 1, 0)


print(f'\ntest prediction:\n{test_prediction}')

test_prediction_list = test_prediction.tolist()
print(test_prediction_list)

with open('predictions2.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(test_prediction_list)



