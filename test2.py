
import numpy as np
import pandas as pd
import sklearn.linear_model as skl_lm
from sklearn.model_selection import GridSearchCV

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 15)  # see up to 15 columns
np.set_printoptions(suppress=True)  # shows 0.07 instead of 7*e^-2
df = pd.read_csv(r'C:\Users\mariu\PycharmProjects\machineLearning\train.csv')  # read in data in a panda dataframe

# Creating an empty 2D Numpy array with two columns
array = np.empty([0, 3])

# Adding elements to the array
array = np.append(array, [[8, 2, 'da']], axis=0)
array = np.append(array, [[2, 2, 'al']], axis=0)
array = np.append(array, [[10, 'hej', 0]], axis=0)

indices = np.argsort(array[:, 0].astype(int))[::-1]
array = array[indices]
print(array)

