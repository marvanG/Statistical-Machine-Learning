import numpy as np
import pandas as pd
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
pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 20)  # see up to 15 columns
pd.set_option('display.max_rows', 3000)  # see up to 15 columns
np.set_printoptions(suppress=True)  # shows 0.07 instead of 7*e^-2

# Data preparation
df = pd.read_csv(r'C:\Users\mariu\PycharmProjects\machineLearning\train.csv')  # read in data in a panda dataframe

# General information
# print(data_frame)
# print(df.info())
# print(data_frame.describe())
# print(data_frame.shape)

# create dummy age diff
lead_age = np.array(df['Age Lead'])
co_lead_age = np.array(df['Age Co-Lead'])
age_difference = lead_age - co_lead_age
prediction = np.where(age_difference < 0, 1, 2)

y = df['Lead']
df = df.drop('Lead', axis=1)

df['Age Difference'] = prediction
df = pd.get_dummies(df, columns=['Age Difference'], prefix='AgeDiff')
df['Lead'] = y

# Drop for now
#df = df.drop(['AgeDiff_2', 'Year', 'Gross', 'AgeDiff_1'], axis=1)
# df = df[['Number of female actors', 'Number of male actors', 'Lead']]

# Analyse data
single_gender_actor_movies = df[(df['Number of female actors'] == 1) | (df['Number of male actors'] == 1)]
two_gender_actor_movies = df[(df['Number of female actors'] == 2) | (df['Number of male actors'] == 2)]
one_female = df[(df['Number of female actors'] == 1)]
one_male = df[(df['Number of male actors'] == 1)]
two_female = df[(df['Number of female actors'] == 2)]
two_male = df[(df['Number of male actors'] == 2)]
man_lead = two_gender_actor_movies[(two_gender_actor_movies['Lead'] == 'Male')]
female_lead = two_gender_actor_movies[(two_gender_actor_movies['Lead'] == 'Female')]

# print(two_female[(two_female['Lead'] == 'Female')])



# One actor test
# print(two_females.head(10))
age_male = np.array(single_gender_actor_movies['Mean Age Male'])
age_female = np.array(single_gender_actor_movies['Mean Age Female'])
age_lead = np.array(single_gender_actor_movies['Age Lead'])
y_test = np.array(single_gender_actor_movies['Lead'])
words_male = np.array(single_gender_actor_movies['Number words male'])
words_female = np.array(single_gender_actor_movies['Number words female'])
number_men = np.array(single_gender_actor_movies['Number of male actors'])

prediction = np.where((words_female == 0) | ((words_male != 0) & (number_men == 1)), 'Female', 'Male')
accuracy = np.mean(prediction == y_test)
print(f'accuracy: {accuracy} ')
print(f'Naive accuracy: 0.756 ')
conf_matrix = pd.crosstab(prediction, y_test)
print(f'{conf_matrix}\n')


"""
print('Number of male or female actors = 2\n')
print(two_gender_actor_movies.head(15))
print('\n')
"""

"""
test_df = df.drop(single_gender_actor_movies.index)

# Compare to naive accuracy
m_actors = np.array(test_df['Number of male actors'])
f_actors = np.array(test_df['Number of female actors'])
actor_diff = m_actors-f_actors

y_test = np.array(test_df['Lead'])
prediction = np.where(actor_diff >= 0, 'Male', 'Female')
accuracy = np.mean(prediction == y_test)
print(f'accuracy: {accuracy} ')
print(f'Naive accuracy: 0.756 ')
conf_matrix = pd.crosstab(prediction, y_test)
print(f'{conf_matrix}\n')

"""

# Compare to naive accuracy


test_df = df
single_act = test_df[(test_df['Number of female actors'] == 1) | (test_df['Number of male actors'] == 1)]

test_df = test_df.drop(single_act.index)


# test_df = two_gender_actor_movies
# single_act = two_gender_actor_movies[(two_gender_actor_movies['Number of female actors'] == 1) | (two_gender_actor_movies['Number of male actors'] == 1)]
# test_df = test_df.drop(single_act.index)


words_male = np.array(test_df['Number words male'])
words_female = np.array(test_df['Number words female'])
lead_word = np.array(test_df['Number of words lead'])
word_diff = np.array(test_df['Difference in words lead and co-lead'])
colead_word = lead_word - word_diff

m_actors = np.array(test_df['Number of male actors'])
f_actors = np.array(test_df['Number of female actors'])
actor_diff = m_actors - f_actors

lead_age = np.array(test_df['Age Lead'])
co_lead_age = np.array(test_df['Age Co-Lead'])
age_difference = lead_age - co_lead_age

female_colead_diff = words_female - colead_word  # if negative lead = female
male_colead_diff = words_male - colead_word  # if negative lead = male or maybe <100
avg_male_word = male_colead_diff / (m_actors - 1)
avg_female_word = female_colead_diff / (f_actors - 1)

y_test = np.array(test_df['Lead'])
prediction = np.where(((avg_female_word < 100) | (avg_female_word > colead_word)) | (
            ((actor_diff < 0) & (age_difference < 0)) & (~((avg_male_word < 100) | (avg_male_word >= colead_word)))),
                      'Female', 'Male')

# prediction = np.where(((avg_male_word < 100) | (avg_male_word >= colead_word)), 'Female', 'Male')
accuracy = np.mean(prediction == y_test)





mask2 = (avg_female_word < 100) | (avg_female_word > colead_word)  # Lead female
mask3 = (avg_male_word < 100) | (avg_male_word > colead_word)  # Lead male

subset1 = df[(df['Number of female actors'] == 1) | (df['Number of male actors'] == 1)]
subset2 = test_df[mask2]
subset3 = test_df[mask3]

test_df = test_df.drop(subset2.index)
test_df = test_df.drop(subset3.index)

m_actors = np.array(test_df['Number of male actors'])
f_actors = np.array(test_df['Number of female actors'])
actor_diff = m_actors - f_actors
lead_age = np.array(test_df['Age Lead'])
co_lead_age = np.array(test_df['Age Co-Lead'])
age_difference = lead_age - co_lead_age
mask4 = (actor_diff < -1) & (age_difference < -1) # Lead probably a female, 0.97%. Might be overfitting. Try with and without it. create a dummy instead?

subset4 = test_df[mask4]
test_df = test_df.drop(subset4.index)

words_male = np.array(single_gender_actor_movies['Number words male'])
words_female = np.array(single_gender_actor_movies['Number words female'])
number_men = np.array(single_gender_actor_movies['Number of male actors'])

predictions1 = np.where((words_female == 0) | ((words_male != 0) & (number_men == 1)), 'Female', 'Male')
predictions2 = np.full(subset2.shape[0], 'Female')
predictions3 = np.full(subset3.shape[0], 'Male')
predictions4 = np.full(subset4.shape[0], 'Female')

removed_count = predictions3.shape[0] + predictions2.shape[0] +predictions1.shape[0] +predictions4.shape[0]
print(f'{predictions1.shape[0]} + {predictions2.shape[0]} + {predictions3.shape[0]} + {predictions4.shape[0]} = {removed_count}')


print(test_df.shape)
print('total: ')
print(removed_count+test_df.shape[0])

print(test_df.describe())
# testing

"""
m_actors = np.array(df['Number of male actors'])
f_actors = np.array(df['Number of female actors'])
actor_diff = m_actors - f_actors

lead_age = np.array(df['Age Lead'])
co_lead_age = np.array(df['Age Co-Lead'])
age_difference = lead_age - co_lead_age

y_test = np.array(df['Lead'])

prediction = np.where((actor_diff < -1) & (age_difference < -1), 'Female', 'Male')

accuracy = np.mean(prediction == y_test)

"""

"""
false_positive_rows = test_df[(prediction == 'Male') & (y_test == 'Female')]
false_negative_rows = test_df[(prediction == 'Female') & (y_test == 'Male')]
print(false_positive_rows)
print(false_negative_rows)
"""
"""
print(f'\naccuracy: {accuracy} ')
print(f'Naive accuracy: 0.756 ')
conf_matrix = pd.crosstab(prediction, y_test)
print(f'{conf_matrix}\n')
"""