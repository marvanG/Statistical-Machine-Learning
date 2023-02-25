import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
pd.set_option('display.width', 200)
pd.set_option('display.max_columns',15)
np.set_printoptions(suppress=True)
url = 'https://uu-sml.github.io/course-sml-public/data/biopsy.csv'
biopsy = pd.read_csv(url, na_values='?', dtype={'ID': str}).dropna().reset_index()

print(biopsy.head())
print(biopsy.info())
print(biopsy.describe())
pd.plotting.scatter_matrix(biopsy.iloc[:, 2:11], figsize=(10,10))
#plt.show()
np.random.seed(1)
N = biopsy.shape[0]
n = 300
rnd_Index = np.random.choice(N, n, replace= False)
rnd_bool = biopsy.index.isin(rnd_Index)
train = biopsy.iloc[rnd_bool]
test = biopsy.iloc[~rnd_bool]


x_cols = ['V3', 'V4', 'V5']
y_cols = 'class'
# Logistic regression
model = skl_lm.LogisticRegression(solver='lbfgs')
X_train = train[x_cols]
Y_train = train[y_cols]
X_test = test[x_cols]
Y_test = test[y_cols]

model.fit(X_train, Y_train)

pred_prob = model.predict_proba(X_test)
print('Class order in the model: ')
print(model.classes_)
print('First 5 predictions: ')
print(pred_prob[0:5])

prediction = np.empty(len(X_test), dtype=object)
prediction = np.where(pred_prob[:, 0]>= 0.5, 'benign', 'malignant')
print(prediction[0:5])
#Accuracy and confusion matrix
accuracy = np.mean(prediction== Y_test)
print(f'Accuracy: {accuracy}')
print('Confusion matrix: \n')

print(pd.crosstab(prediction, Y_test))

#LDA
model = skl_da.LinearDiscriminantAnalysis()
model.fit(X_train,Y_train)

predict_prob = model.predict_proba(X_test)
print('Class order in model: ')
print(model.classes_)

print(f'First five probabilities:\n {predict_prob[0:5]}')

prediction = np.empty(len(X_test), dtype=object)
prediction = np.where(predict_prob[:, 0]>= 0.5, 'benign', 'malignant')
print(prediction[0:5])

#Accuracy and confusion matrix
accuracy = np.mean(prediction==Y_test)
print(f'Accuracy: {accuracy}')
print('Confusion matrix: \n')
print(pd.crosstab(prediction, Y_test))


#QDA
model = skl_da.QuadraticDiscriminantAnalysis()
model.fit(X_train,Y_train)


predict_prob = model.predict_proba(X_test)
print('Class order in model: ')
print(model.classes_)

print(f'First five probabilities:\n {predict_prob[0:5]}')

prediction = np.empty(len(X_test), dtype=object)
prediction = np.where(predict_prob[:, 0]>= 0.5, 'benign', 'malignant')
print(prediction[0:5])

#Accuracy and confusion matrix
accuracy = np.mean(prediction==Y_test)
print(f'Accuracy: {accuracy}')
print('Confusion matrix: \n')
print(pd.crosstab(prediction, Y_test))
