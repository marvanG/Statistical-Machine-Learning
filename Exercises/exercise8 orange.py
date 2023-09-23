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

sklearn.set_config(display='full')
# Nice options
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 15)  # see up to 15 columns
np.set_printoptions(suppress=True)  # shows 0.07 instead of 7*e^-2

OJ = pd.read_csv('https://uu-sml.github.io/course-sml-public/data/oj.csv')

# Sampling

np.random.seed(1)

n = 800
N = OJ.shape[0]
TrainIndex = np.random.choice(N, n, replace=False)

trainbool = OJ.index.isin(TrainIndex)
train = OJ.iloc[trainbool]
test = OJ.iloc[~trainbool]

x_train = train.drop('Purchase', axis=1)
x_train = pd.get_dummies(x_train, columns=['Store7'])
y_train = train['Purchase']



model = tree.DecisionTreeClassifier(max_depth=2)


model.fit(x_train, y_train)
#print(OJ.info())

dot_data = tree.export_graphviz(
    model, out_file=None, feature_names=x_train.columns,
    class_names=model.classes_, filled=True, rounded=True,
    leaves_parallel=True, proportion=True)

graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render('tree')

image = imread('tree.png')

plt.imshow(image)
plt.axis('off')


x_test = test.drop('Purchase', axis=1)
x_test = pd.get_dummies(x_test, columns=['Store7'])
y_test = test['Purchase']
y_predict = model.predict(x_test)
conf_matrix = pd.crosstab(y_predict, y_test)
accuracy = np.mean(y_predict == y_test)
print(f'Accuracy = {accuracy}')
print(f'confusion matrix:\n{conf_matrix}')



plt.show()