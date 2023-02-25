import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

# Nice options
pd.set_option('display.width', 200)
pd.set_option('display.max_columns',15) # see up to 15 columns
np.set_printoptions(suppress=True)  # shows 0.07 instead of 7*e^-2

# Data preparation
data_frame = pd.read_csv(r'C:\Users\mariu\PycharmProjects\machineLearning\train.csv') #read in data in a panda dataframe
# General information
# print(data_frame)
print(data_frame.info())
# print(data_frame.describe())
# print(data_frame.shape)


# Scatter plot, to play with data analysis and check for interesting correlations
# x and y are column index. Use print(data_frame.info()) to check indexes
"""
x = 0
y = 10
fig, ax = plt.subplots()
ax.scatter(data_frame.iloc[:,x], data_frame.iloc[:,y])
ax.set_xlabel(data_frame.columns[x])
ax.set_ylabel(data_frame.columns[y])
plt.show()
"""

# Set the random seed
np.random.seed(1)

# Shuffle the rows of the dataframe
data_frame = data_frame.sample(frac=1)

# Normalization
"""
Y = data_frame['Lead']
X_df = data_frame.drop(['Lead'], axis=1)
df_values = X_df.values
# Normalize the numpy array
norm_df_values = skl_pre.normalize(df_values)
# Create a new dataframe from the normalized array
norm_data_frame = pd.DataFrame(norm_df_values, columns=X_df.columns)
norm_data_frame['Lead'] = Y
print(norm_data_frame.head())
"""


# Calculate the number of datapoints per fold
n = 10
fold_size = data_frame.shape[0] // n
accuracy_list = []
# Split the dataframe into 10 folds
for i in range(n):
    start = i * fold_size
    end = start + fold_size
    X_test = data_frame.iloc[start:end]
    X_train = data_frame.drop(X_test.index)
    Y_test = X_test['Lead']
    Y_train = X_train['Lead']
    X_test = X_test.drop(['Lead'], axis = 1)
    X_train = X_train.drop(['Lead'], axis = 1)
    # Use X_train and X_test as the training and validation sets, respectively

    # Logistic regression model
    model = skl_lm.LogisticRegression(solver='liblinear')
    model.fit(X_train, Y_train)
    print(f'Model: {i}')

    # prediction, probability
    pred_prob = model.predict_proba(X_test)
    print('Class order in this model: ')
    print(f'{model.classes_} \n')

    print('First 5 probability predictions: ')
    print(f'{pred_prob[0:5]} \n')

    # prediction
    prediction = np.empty(len(X_test), dtype=object)
    prediction = np.where(pred_prob[:, 0] >= 0.5, 'Female', 'Male')
    print('First 5 predictions: ')
    print(f'{prediction[0:5]} ')

    # Accuracy and confusion matrix
    accuracy = np.mean(prediction == Y_test)
    conf_matrix = pd.crosstab(prediction, Y_test)
    print('Confusion matrix: ')
    print(f'{conf_matrix} ')
    print(f'Accuracy: {accuracy} ')
    accuracy_list.append(accuracy)

    # Naive classifier
    zeros = np.zeros((len(X_test)))
    print(len(X_test))
    naive_prediction = np.where(zeros == 0, 'Male','')
    naive_accuracy = np.mean(naive_prediction == Y_test)
    print(f'Naive accuracy: {naive_accuracy} \n')





avg_accuracy = np.mean(accuracy_list)
print(accuracy_list)
print(f'Mean accuracy: {avg_accuracy}')
"""
# Logistic regression model
X_test = data_frame.iloc[0:500]
X_train = data_frame.drop(X_test.index)
Y_test = X_test['Lead']
Y_train = X_train['Lead']
X_test = X_test.drop(['Lead'], axis = 1)
X_train = X_train.drop(['Lead'], axis = 1)

print(X_train)
print(Y_test)
model = skl_lm.LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)


# prediction, probability
pred_prob = model.predict_proba(X_test)
print('Class order in this model: ')
print(f'{model.classes_} \n')

print('First 5 probability predictions: ')
print(f'{pred_prob[0:5]} \n')

# prediction
prediction = np.empty(len(X_test), dtype=object)
prediction = np.where(pred_prob[:, 0]>= 0.5, 'Female', 'Male')
print('First 5 predictions: ')
print(f'{prediction[0:5]} \n')

# Accuracy and confusion matrix
accuracy = np.mean(prediction == Y_test)
conf_matrix = pd.crosstab(prediction, Y_test)
print('Confusion matrix: ')
print(f'{conf_matrix} \n')
print(f'Accuracy: {accuracy}')

"""








