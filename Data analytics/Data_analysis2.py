from random import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

# Nice options
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 300)  # see up to 15 columns
np.set_printoptions(suppress=True)  # shows 0.07 instead of 7*e^-2

# Data preparation
df = pd.read_csv(r'C:\Users\mariu\PycharmProjects\machineLearning\train.csv')  # read in data in a panda dataframe
# print(df.info())

print()

lead_age = np.array(df['Age Lead'])
co_lead_age = np.array(df['Age Co-Lead'])
age_difference = lead_age - co_lead_age
prediction = np.where(age_difference < 0, 1, 2)

y = df['Lead']
df = df.drop('Lead', axis=1)

df['Age Difference'] = prediction
df = pd.get_dummies(df, columns=['Age Difference'], prefix='AgeDiff')
df['Lead'] = y

##############

year = np.array(df['Year'])
gross = np.array(df['Gross'])

prediction = np.where(gross < 200, 1, 2)

y = df['Lead']
df = df.drop('Lead', axis=1)

df['Age Difference'] = prediction
df = pd.get_dummies(df, columns=['Age Difference'], prefix='AgeDiff')
df['Lead'] = y

print(df.info())

"""


prediction = np.where(age_difference >= 0, 'Male', 'Male')
accuracy = np.mean(prediction == y_test)
print(f'accuracy: {accuracy} ')
conf_matrix = pd.crosstab(prediction, y_test)
print(f'{conf_matrix}\n')
"""
