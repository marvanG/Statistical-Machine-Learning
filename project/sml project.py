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
pd.set_option('display.max_rows', 60)  # see up to 15 columns
np.set_printoptions(suppress=True)  # shows 0.07 instead of 7*e^-2

# Data preparation
origin_df = pd.read_csv(
    r'C:\Users\mariu\PycharmProjects\machineLearning\train.csv')  # read in data in a panda dataframe



df = origin_df

clasi = df['Lead']
df = df.drop('Lead', axis=1)
df['lead_colead_age_diff'] = df['Age Lead'] - df['Age Co-Lead']
df['Lead'] = clasi

# Set the random seed
#np.random.seed(23)
#df = df.sample(frac=1)

df_test = df.iloc[0:240]
y_safety = df_test['Lead']

print(df_test['Lead'].head(10))
#df = df.drop(df_test.index)


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
# split into subsets with math


test_df = df
single_act = test_df[(test_df['Number of female actors'] == 1) | (test_df['Number of male actors'] == 1)]
test_df = test_df.drop(single_act.index)

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

"""
y_test = np.array(test_df['Lead'])
prediction = np.where(((avg_female_word < 100) | (avg_female_word > colead_word)) | (
            ((actor_diff < 0) & (age_difference < 0)) & (~((avg_male_word < 100) | (avg_male_word >= colead_word)))),
                      'Female', 'Male')

# prediction = np.where(((avg_male_word < 100) | (avg_male_word >= colead_word)), 'Female', 'Male')
accuracy = np.mean(prediction == y_test)
print(accuracy)
"""

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

mask4 = (actor_diff < 0) & (
        age_difference < -1)  # Lead probably a female, 0.97%. Might be overfitting. Try with and without it. create a dummy instead?
subset4 = test_df[mask4]
test_df = test_df.drop(subset4.index)

words_male = np.array(single_gender_actor_movies['Number words male'])
words_female = np.array(single_gender_actor_movies['Number words female'])
number_men = np.array(single_gender_actor_movies['Number of male actors'])

predictions1 = np.where((words_female == 0) | ((words_male != 0) & (number_men == 1)), 'Female', 'Male')
predictions2 = np.full(subset2.shape[0], 'Female')
predictions3 = np.full(subset3.shape[0], 'Male')
predictions4 = np.full(subset4.shape[0], 'Female')

removed_count = predictions3.shape[0] + predictions2.shape[0] + predictions1.shape[0] + predictions4.shape[0]
print(
    f'{predictions1.shape[0]} + {predictions2.shape[0]} + {predictions3.shape[0]} + {predictions4.shape[0]} = {removed_count}')

print(f'{test_df.shape[0]} + {removed_count} = {removed_count + test_df.shape[0]}')
print('total: ')
print(removed_count + test_df.shape[0])

# testing
classifier = test_df['Lead']
z = np.zeros(test_df.shape[0])
prediction = np.where(z == 0, 'Male', 'Male')
accuracy = np.mean(prediction == classifier)

"""
false_positive_rows = test_df[(prediction == 'Male') & (y_test == 'Female')]
false_negative_rows = test_df[(prediction == 'Female') & (y_test == 'Male')]
print(false_positive_rows)
print(false_negative_rows)
"""

print(f'\nNew naive accuracy: {accuracy} ')
print(f'Old naive accuracy: 0.756 ')
conf_matrix = pd.crosstab(prediction, classifier)
print(f'{conf_matrix}\n')

seeds = 1  # How many cross validations to run (more than 5 takes alot of time)
cut_off = 0.71
df = test_df




# Example of argument cut_off_accuracy: 0.8 stops the algorithm if accuracy gets below 0.8
# cut_off_accuracy = 0: Stops when all features have been removed.
def feature_dropping(d_f, model, cut_off_accuracy):
    feature_list = np.array(d_f.columns)
    data_f = d_f
    iteration = 0
    max_accuracy_cols = np.array(df.columns)
    dropped_cols = np.array([])
    max_dropped_cols = np.array([])

    # calculate accuracy with all features.
    accuracy = multiple_k_fold_cross_validation(d_f, model, seeds)
    print(f'Accuracy with all features: {accuracy}\n')
    global_max_accuracy = accuracy

    # Dropping one feature at a time
    while len(feature_list) > 2:
        iteration += 1
        local_max_accuracy = 0

        for i in range(len(feature_list) - 1):

            accuracy = multiple_k_fold_cross_validation(df_with_drop(data_f, feature_list[i]), model, seeds)

            if accuracy >= local_max_accuracy:
                local_max_accuracy = accuracy
                col_to_drop = i
            elif accuracy == local_max_accuracy:
                print(f'Acuracy for dropping {feature_list[i]} == as dropping {feature_list[col_to_drop]}')

        print(f'Iteration: {iteration}')

        col = feature_list[col_to_drop]
        dropped_cols = np.append(dropped_cols, col)

        print(f'Column to drop is: {col}')
        print(f'Accuracy = {local_max_accuracy} ')

        data_f = data_f.drop(feature_list[col_to_drop], axis=1)
        feature_list = np.array(data_f.columns)
        print(f'{len(feature_list[:-1])} features left: \n{feature_list[:-1]}\n')
        if local_max_accuracy < cut_off_accuracy:
            print(f'Cut off accuracy {cut_off_accuracy} reached at iteration {iteration}.\n')
            break

        if local_max_accuracy >= global_max_accuracy:
            global_max_accuracy = local_max_accuracy
            max_accuracy_cols = feature_list
            max_dropped_cols = dropped_cols

    print('\nFinished\n')
    print(
        f'Best accuracy = {global_max_accuracy} with {len(max_accuracy_cols[:-1])} features: \n{max_accuracy_cols[:-1]}\n\n{len(max_dropped_cols)} Features dropped: {max_dropped_cols}')
    return global_max_accuracy


def df_with_drop(data_frame, col):
    data_frame = data_frame.drop(col, axis=1)
    return data_frame


def multiple_k_fold_cross_validation(data_frame, model, n):
    avg_accuracy_list = []
    for seed in range(n):
        avg_accuracy = k_fold_cross_validation(data_frame, model, seed)
        avg_accuracy_list.append(avg_accuracy)

    total_mean_accuracy = np.mean(avg_accuracy_list)
    return total_mean_accuracy


def k_fold_cross_validation(data_frame, model, seed):
    # Set the random seed
    np.random.seed(seed + 15)

    # Shuffle the rows of the dataframe
    data_frame = data_frame.sample(frac=1)
    # Calculate the number of datapoints per fold
    n = 5
    fold_size = data_frame.shape[0] // n
    acc_list = []

    # Split the dataframe into 10 folds
    for j in range(n):
        start = j * fold_size
        end = start + fold_size
        x_test = data_frame.iloc[start:end]
        x_train = data_frame.drop(x_test.index)
        y_test = x_test['Lead']
        y_train = x_train['Lead']
        x_test = x_test.drop(['Lead'], axis=1)
        x_train = x_train.drop(['Lead'], axis=1)
        # Use x_train and x_test as the training and validation sets, respectively

        accuracy = accuracy_calculator(model, x_train, y_train, x_test, y_test)
        acc_list.append(accuracy)

    avg_accuracy = np.mean(acc_list)
    return avg_accuracy


def accuracy_calculator(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    # prediction, probability
    prediction_probability = model.predict_proba(x_test)
    prediction = np.empty(len(x_test), dtype=object)
    prediction = np.where(prediction_probability[:, 0] > 0.5, 'Female', 'Male')
    # Accuracy and confusion matrix
    accuracy = np.mean(prediction == y_test)
    conf_matrix = pd.crosstab(prediction, y_test)
    # print(f'confusion matrix:\n{conf_matrix}')
    return accuracy


def compare_models(model_array, data_frames):
    for i in range(len(model_array)):
        model = model_array[i, 0]
        print(f'\nNow running: {model_array[i, 1]}\n')
        accuracy = feature_dropping(data_frames, model, cut_off)
        model_array[i, 2] = accuracy

    indices = np.argsort(model_array[:, 2].astype(float))[::-1]
    sorted_models = model_array[indices]
    return sorted_models

    # Print results


def compare_norms_and_models(model_array, dfs):  # arguments are an array with stored models and an array of
    # dataframes(original, normalized etc)

    best_accuracy = 0
    best_norm = ''
    best_model = ''
    results = []
    names = []
    for df_norm, name in dfs:
        print(f'\nNow running: {name}\n')
        result = compare_models(model_array, df_norm)
        results.append(result)
        names.append(name)

        if result[0, 2] >= best_accuracy:
            best_accuracy = result[0, 2]
            best_model = result[0, 1]
            best_norm = name
        if result[0, 2] == best_accuracy:
            print(f'Same results for {name} and {result[0, 1]} as for {best_norm} and {best_model}\n')

    for i in range(len(results)):
        print(f'\nResults in for {names[i]}')
        print_results(results[i])
    print(f'\nResults:\n')
    print(f'Best norm: {best_norm}')
    print(f'Best model: {best_model}')
    print(f'Accuracy = {best_accuracy}')


def print_results(sorted_models):
    print('List of models ordered by accuracy:\n')
    for i in range(len(sorted_models)):
        print(f'{i + 1}. {sorted_models[i, 1]} accuracy = {sorted_models[i, 2]}')

    print(f'\nBest method: {sorted_models[0, 1]}')
    print(f'Accuracy = {sorted_models[0, 2]}\n')


# Naive classifier (Only guess male)
zeros = np.zeros((df.shape[0]))
naive_prediction = np.where(zeros == 0, 'Male', '')
y_naive_test = np.array(df['Lead'])
naive_accuracy = np.mean(naive_prediction == y_naive_test)
print(f'New naive accuracy: {naive_accuracy}')

# normalizing techniques
# Drop classifier Lead
# df = df.reset_index(drop=True)
classifier = df['Lead']

df_minus_classifier = df.drop('Lead', axis=1)

# Basic normalization method:
df_norm_1 = (df_minus_classifier - df_minus_classifier.mean()) / df_minus_classifier.std()
df_norm_1['Lead'] = classifier

# MinMaxScaler:
min_max_scaler = preprocessing.MinMaxScaler()
df_min_max = pd.DataFrame(min_max_scaler.fit_transform(df_minus_classifier), columns=df_minus_classifier.columns)
df_min_max['Lead'] = classifier

# RobustScaler:
robust_scaler = preprocessing.RobustScaler()
df_robust = pd.DataFrame(robust_scaler.fit_transform(df_minus_classifier), columns=df_minus_classifier.columns)
df_robust['Lead'] = classifier

# StandardScaler:
standard_scaler = preprocessing.StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df_minus_classifier), columns=df_minus_classifier.columns)
df_standard['Lead'] = classifier

# Store normalized dataframes (use to test normalization in compare_norms_and_models()


# df_array = [[df, 'Original df'], [df_robust, 'Robust Scaler']]


# [df_standard, 'Standard Scaler']

# df_array = [[df_standard, 'Standard Scaler']]
# df_array = [[df_min_max, 'Min Max Norm']]
# df_array = [[df_robust, 'Robust Scaler']]
# only standard, saves time and normalization sucked for Log Reg
df_array = [[df, 'Original df']]

# Create models
# Logistic regression model
# model1 = skl_lm.LogisticRegression(solver='liblinear', penalty='l2', C=0.1)
model1 = skl_lm.LogisticRegression(solver='liblinear')

# Linear discriminant analysis
model2 = skl_da.LinearDiscriminantAnalysis()

# Quadratic discriminant analysis
model3 = skl_da.QuadraticDiscriminantAnalysis()

# Classification tree
model4 = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=10)
# model4 = tree.DecisionTreeClassifier()

# Bagging tree classifier
decision_tree_base_estimator = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
model11 = BaggingClassifier()
model5 = BaggingClassifier(estimator=decision_tree_base_estimator)

# Random forest classifier
# model6 = RandomForestClassifier(max_depth=10, min_samples_split=6, n_estimators=200)
model6 = RandomForestClassifier()

# KNN
model7 = KNeighborsClassifier()

# AdaBoosting

model8 = AdaBoostClassifier()

# Gradient Boosting

model9 = GradientBoostingClassifier(subsample=0.5, min_samples_split=25, n_estimators=200)
model10 = GradientBoostingClassifier()
# model9 = GradientBoostingClassifier()


# Store models (to run comparisons)
models = np.empty([0, 3])
models = np.append(models, [[model1, 'Logistic regression', 0]], axis=0)
models = np.append(models, [[model2, 'linear discriminant analysis', 0]], axis=0)
models = np.append(models, [[model3, 'Quadratic discriminant analysis', 0]], axis=0)
# models = np.append(models, [[model4, 'Decision tree classifier', 0]], axis=0)
models = np.append(models, [[model5, 'Bagging tree classifier modified', 0]], axis=0)
models = np.append(models, [[model6, 'Random forest classifier', 0]], axis=0)
models = np.append(models, [[model7, 'KNN', 0]], axis=0)
models = np.append(models, [[model8, 'AdaBoosting classifier', 0]], axis=0)
models = np.append(models, [[model9, 'Gradient boosting classifier modified', 0]], axis=0)
models = np.append(models, [[model10, 'Gradient boosting classifier default', 0]], axis=0)
models = np.append(models, [[model11, 'Bagging tree classifier default', 0]], axis=0)
# Run test, find the best normalization and features.

# compare_norms_and_models(models, df_array)

"""
# Logistic Regression
# ['Gross' 'Mean Age Female']
# Number words male
# Parameter optimization
# X and Y

train_acc_list = np.array([])
score_list = np.array([])
test_acc_list = np.array([])
for i in range(10):
    np.random.seed(i)
    np.random.seed(i)
    # Shuffle the rows of the dataframe
    df_shuffle = df.sample(frac=1)
    X_test = df_shuffle.iloc[0:120]
    X_train = df_shuffle.drop(X_test.index)
    Y_test = X_test['Lead']
    Y_train = X_train['Lead']
    X_test = X_test.drop(['Lead'], axis=1)
    X_train = X_train.drop(['Lead'], axis=1)
    # Logistic regression
    # Optimizing Log Reg using grid search (built-in function that tests combinations of hyperparameters)

    X_test = X_test.drop(['Mean Age Female', 'Difference in words lead and co-lead', 'Gross', 'Year',
 'Number of words lead', 'lead_colead_age_diff', 'Age Lead', 'Mean Age Male',
 'Number words male', 'Number words female'], axis=1)
    X_train = X_train.drop(['Mean Age Female', 'Difference in words lead and co-lead', 'Gross', 'Year',
 'Number of words lead', 'lead_colead_age_diff', 'Age Lead', 'Mean Age Male',
 'Number words male', 'Number words female'], axis=1)

    # parameters, every combination will be tested for the best result.
    # 'C': np.logspace(-2, 2, 30)
    # 'C': [0.01, 0.1, 0.04, 0.3],
    # 'C': np.logspace(-2, 2, 30),
    param_grid = [
        {'penalty': ['l2', 'l1'],
         'C': np.logspace(-2, 2, 30),
         'solver': ['liblinear'],
         'max_iter': [100],
         }
    ]
    logModel = skl_lm.LogisticRegression()
    clf = skl_ms.GridSearchCV(logModel, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    grid_search_result = clf.fit(X_train, Y_train)

    resulting_model = grid_search_result.best_estimator_
    best_parameters = grid_search_result.best_params_

    print(f'\nGrid search complete for Logistic regression. Results:')
    print(best_parameters)
    print(f'Train accuracy = {resulting_model.score(X_train, Y_train)}')
    print(f'score:{clf.best_score_}')
    print(f'Test accuracy: {resulting_model.score(X_test, Y_test)}\n')
    train_acc_list = np.append(train_acc_list, resulting_model.score(X_train, Y_train))
    score_list = np.append(score_list, clf.best_score_)
    test_acc_list = np.append(test_acc_list, resulting_model.score(X_test, Y_test))

print(f'Test over, results:\n')
print(f'Average train accuracy = {np.mean(train_acc_list)}')
print(f'Average score = {np.mean(score_list)}')
print(f'Average test accuracy = {np.mean(test_acc_list)}')
"""
"""
# Decision tree classifier
train_acc_list = np.array([])
score_list = np.array([])
test_acc_list = np.array([])
for i in range(10):
    np.random.seed(i)
    # Shuffle the rows of the dataframe
    df_shuffle = df_standard.sample(frac=1)
    # 7 Features dropped: ['Gross' 'Year' 'Mean Age Male' 'Age Co-Lead' 'Mean Age Female' 'Age Lead'
    #  'Total words']
    X_test = df_shuffle.iloc[0:200]
    X_train = df_shuffle.drop(X_test.index)
    Y_test = X_test['Lead']
    Y_train = X_train['Lead']
    X_test = X_test.drop(['Lead'], axis=1)
    X_train = X_train.drop(['Lead'], axis=1)

    X_test = X_test.drop(
        ['Gross', 'Year', 'Mean Age Male', 'Age Co-Lead', 'Mean Age Female', 'Age Lead', 'Total words'], axis=1)
    X_train = X_train.drop(
        ['Gross', 'Year', 'Mean Age Male', 'Age Co-Lead', 'Mean Age Female', 'Age Lead', 'Total words'], axis=1)

    # parameters, every combination will be tested for the best result.
    param_grid = {'max_depth': [10],
                  'criterion': ['entropy'],
                  #'min_samples_split': [10],


                  }

    decisionTreeModel = tree.DecisionTreeClassifier()
    DT_grid = skl_ms.GridSearchCV(decisionTreeModel, param_grid=param_grid, cv=10, verbose=True, n_jobs=-1)
    grid_search_result = DT_grid.fit(X_train, Y_train)

    resulting_model = grid_search_result.best_estimator_
    best_parameters = grid_search_result.best_params_

    print('\n')
    print(f'Grid search complete for Decision tree classifier. Results:\n')
    print(resulting_model)
    print(best_parameters)
    print(f'Train accuracy = {resulting_model.score(X_train, Y_train)}')
    print(f'score:{DT_grid.best_score_}')
    print(f'Test accuracy: {resulting_model.score(X_test, Y_test)}')
    train_acc_list = np.append(train_acc_list, resulting_model.score(X_train, Y_train))
    score_list = np.append(score_list, DT_grid.best_score_)
    test_acc_list = np.append(test_acc_list, resulting_model.score(X_test, Y_test))

print(f'Test over, results:\n')
print(f'Average train accuracy = {np.mean(train_acc_list)}')
print(f'Average score = {np.mean(score_list)}')
print(f'Average test accuracy = {np.mean(test_acc_list)}')
"""
"""
DT_model = tree.DecisionTreeClassifier(max_depth=5, max_features='auto', random_state=1, criterion='gini', min_samples_split=2)
DT_model.fit(X_train,Y_train)

dot_data = tree.export_graphviz(
    DT_model, out_file=None, feature_names=X_train.columns,
    class_names=DT_model.classes_, filled=True, rounded=True,
    leaves_parallel=True, proportion=True)

graph = graphviz.Source(dot_data)
graph.format = 'pdf'
graph.render('DecisionTree')
"""
"""
# Bagging classifier

train_acc_list = np.array([])
score_list = np.array([])
test_acc_list = np.array([])

base1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
base2 = RandomForestClassifier(criterion='entropy', max_depth=15, min_samples_split=5)
base3 = skl_lm.LogisticRegression(penalty='l2', C=0.1, solver='liblinear', max_iter=100)
base4 = skl_da.QuadraticDiscriminantAnalysis()
for i in range(10):
    np.random.seed(i)
    # Shuffle the rows of the dataframe
    df_shuffle = df_robust.sample(frac=1)

    X_test = df_shuffle.iloc[0:200]
    X_train = df_shuffle.drop(X_test.index)
    Y_test = X_test['Lead']
    Y_train = X_train['Lead']
    X_test = X_test.drop(['Lead'], axis=1)
    X_train = X_train.drop(['Lead'], axis=1)

    # Log Reg drop
    # X_test = X_test.drop(['Gross', 'Mean Age Female', 'Number words male'], axis=1)
    # X_train = X_train.drop(['Gross', 'Mean Age Female', 'Number words male'], axis=1)

    # QDA drop
    # X_test = X_test.drop(['Gross', 'Year', 'Number words male', 'AgeDiff_1'], axis=1)
    # X_train = X_train.drop(['Gross', 'Year', 'Number words male', 'AgeDiff_1'], axis=1)

    X_test = X_test.drop(['AgeDiff_1', 'Number words male', 'Gross', 'Year', 'Age Lead', 'Age Co-Lead', 'Mean Age Female','Mean Age Male'], axis=1)
    X_train = X_train.drop(['AgeDiff_1', 'Number words male', 'Gross', 'Year', 'Age Lead', 'Age Co-Lead', 'Mean Age Female','Mean Age Male'], axis=1)



    # random forest drop
    # X_test = X_test.drop(['Mean Age Male', 'Gross', 'Year', 'Age Lead', 'Mean Age Female', 'Total words', 'Age Co-Lead'], axis=1)
    # X_train = X_train.drop(['Mean Age Male', 'Gross', 'Year', 'Age Lead', 'Mean Age Female', 'Total words', 'Age Co-Lead'], axis=1)

    # decision tree drop
    # X_test = X_test.drop(['Gross', 'Mean Age Male', 'Year', 'Total words', 'Mean Age Female','Number words male', 'Age Co-Lead', 'Age Lead'], axis=1)
    # X_train = X_train.drop(['Gross', 'Mean Age Male', 'Year', 'Total words', 'Mean Age Female', 'Number words male', 'Age Co-Lead', 'Age Lead'], axis=1)

    # parameters, every combination will be tested for the best result.
    param_grid = {
        'n_estimators': [50],
        'estimator': [base4],
        #'max_samples': [0.4, 0.6, 0.8, 0.9],
        'max_features': [0.4, 0.6, 0.8,0.9],
        #'bootstrap': [True],
        #'bootstrap_features': [False],
        #'oob_score': [False],
        #'warm_start': [False]

    }
    # base_estimator = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
    BaggingTreeModel = BaggingClassifier()
    BT_grid = skl_ms.GridSearchCV(BaggingTreeModel, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    grid_search_result = BT_grid.fit(X_train, Y_train)

    resulting_model = grid_search_result.best_estimator_
    best_parameters = grid_search_result.best_params_

    print('\n')
    print(f'Grid search complete for Bagging tree classifier. Results:\n')
    print(resulting_model)
    print(best_parameters)
    print(f'Train accuracy = {resulting_model.score(X_train, Y_train)}')
    print(f'score:{BT_grid.best_score_}')
    print(f'Test accuracy: {resulting_model.score(X_test, Y_test)}')
    train_acc_list = np.append(train_acc_list, resulting_model.score(X_train, Y_train))
    score_list = np.append(score_list, BT_grid.best_score_)
    test_acc_list = np.append(test_acc_list, resulting_model.score(X_test, Y_test))

print(f'Test over, results:\n')
print(f'Average train accuracy = {np.mean(train_acc_list)}')
print(f'Average score = {np.mean(score_list)}')
print(f'Average test accuracy = {np.mean(test_acc_list)}')
"""
"""""
# Random forest classifier

train_acc_list = np.array([])
score_list = np.array([])
test_acc_list = np.array([])
for i in range(10):
    np.random.seed(i)
    # Shuffle the rows of the dataframe
    df_shuffle = df.sample(frac=1)

    X_test = df_shuffle.iloc[0:120]
    X_train = df_shuffle.drop(X_test.index)
    Y_test = X_test['Lead']
    Y_train = X_train['Lead']
    X_test = X_test.drop(['Lead'], axis=1)
    X_train = X_train.drop(['Lead'], axis=1)
    X_test = X_test.drop(
        ['Difference in words lead and co-lead', 'Year', 'Age Co-Lead',
         'Number words male', 'Gross', 'Total words', 'Number words female'], axis=1)
    X_train = X_train.drop(
        ['Difference in words lead and co-lead', 'Year', 'Age Co-Lead',
         'Number words male', 'Gross', 'Total words', 'Number words female'], axis=1)

    # parameters, every combination will be tested for the best result.
    param_grid = {
        'criterion': ['entropy'],
        'n_estimators': [100],
        'max_depth': [20],
        'min_samples_split': [5],
        # 'bootstrap': [False],
       # 'warm_start':[False],
       #'oob_score':[False]

    }

    RandomForestModel = RandomForestClassifier()
    RF_grid = skl_ms.GridSearchCV(RandomForestModel, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    grid_search_result = RF_grid.fit(X_train, Y_train)

    resulting_model = grid_search_result.best_estimator_
    best_parameters = grid_search_result.best_params_

    print('\n')
    print(f'Grid search complete for Random forest classifier. Results:\n')
    print(resulting_model)
    print(best_parameters)
    print(f'Train accuracy = {resulting_model.score(X_train, Y_train)}')
    print(f'score:{RF_grid.best_score_}')
    print(f'Test accuracy: {resulting_model.score(X_test, Y_test)}')
    train_acc_list = np.append(train_acc_list, resulting_model.score(X_train, Y_train))
    score_list = np.append(score_list, RF_grid.best_score_)
    test_acc_list = np.append(test_acc_list, resulting_model.score(X_test, Y_test))

print(f'Test over, results:\n')
print(f'Average train accuracy = {np.mean(train_acc_list)}')
print(f'Average score = {np.mean(score_list)}')
print(f'Average test accuracy = {np.mean(test_acc_list)}')

"""
"""
# QDA
train_acc_list = np.array([])
score_list = np.array([])
test_acc_list = np.array([])
for i in range(10):
    np.random.seed(i)
    # Shuffle the rows of the dataframe
    df_shuffle = df.sample(frac=1)

    X_test = df_shuffle.iloc[0:200]
    X_train = df_shuffle.drop(X_test.index)
    Y_test = X_test['Lead']
    Y_train = X_train['Lead']
    X_test = X_test.drop(['Lead'], axis=1)
    X_train = X_train.drop(['Lead'], axis=1)


    X_test = X_test.drop(['AgeDiff_1', 'Number words male', 'Gross', 'Year', 'Age Lead', 'Age Co-Lead', 'Mean Age Female', 'Mean Age Male'], axis=1)
    X_train = X_train.drop(['AgeDiff_1', 'Number words male', 'Gross', 'Year', 'Age Lead', 'Age Co-Lead', 'Mean Age Female', 'Mean Age Male'], axis=1)

    #X_test = X_test.drop(['Number words male', 'Gross', 'Year'], axis=1)
    #X_train = X_train.drop(['Number words male', 'Gross', 'Year'], axis=1)

    #X_test = X_test.drop(['Number words male', 'Gross', 'Year', 'AgeDiff_2'], axis=1)
    #X_train = X_train.drop(['Number words male', 'Gross', 'Year', 'AgeDiff_2'], axis=1)

    #X_test = X_test.drop(['Number words male', 'Gross', 'Year', 'AgeDiff_1'], axis=1)
    #X_train = X_train.drop(['Number words male', 'Gross', 'Year', 'AgeDiff_1'], axis=1)

    # parameters, every combination will be tested for the best result.
    param_grid = {
        'reg_param': [0.1],

    }

    QDAmodel = skl_da.QuadraticDiscriminantAnalysis()
    QDA_grid = skl_ms.GridSearchCV(QDAmodel, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    grid_search_result = QDA_grid.fit(X_train, Y_train)

    resulting_model = grid_search_result.best_estimator_
    best_parameters = grid_search_result.best_params_

    print('\n')
    print(f'Grid search complete for QDA. Results:\n')
    print(resulting_model)
    print(best_parameters)
    print(f'Train accuracy = {resulting_model.score(X_train, Y_train)}')
    print(f'score:{QDA_grid.best_score_}')
    print(f'Test accuracy: {resulting_model.score(X_test, Y_test)}')
    train_acc_list = np.append(train_acc_list, resulting_model.score(X_train, Y_train))
    score_list = np.append(score_list, QDA_grid.best_score_)
    test_acc_list = np.append(test_acc_list, resulting_model.score(X_test, Y_test))

print(f'Test over, results:\n')
print(f'Average train accuracy = {np.mean(train_acc_list)}')
print(f'Average score = {np.mean(score_list)}')
print(f'Average test accuracy = {np.mean(test_acc_list)}')
"""
"""

# KNN
train_acc_list = np.array([])
score_list = np.array([])
test_acc_list = np.array([])
for i in range(10):
    np.random.seed(i)
    # Shuffle the rows of the dataframe
    df_shuffle = df_robust.sample(frac=1)

    X_test = df_shuffle.iloc[0:200]
    X_train = df_shuffle.drop(X_test.index)
    Y_test = X_test['Lead']
    Y_train = X_train['Lead']
    X_test = X_test.drop(['Lead', 'Mean Age Male', 'Age Lead', 'Number of words lead', 'Age Co-Lead'], axis=1)
    X_train = X_train.drop(['Lead', 'Mean Age Male', 'Age Lead', 'Number of words lead', 'Age Co-Lead'], axis=1)

    # Create a parameter grid to search over
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 20],

        'algorithm': ['auto'],
        'p': [1, 2],

        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Create a KNN classifier object
    knn = KNeighborsClassifier()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, Y_train)

    resulting_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_

    print('\n')
    print(f'Grid search complete for KNN. Results:\n')
    print(resulting_model)
    print(best_parameters)
    print(f'Train accuracy = {resulting_model.score(X_train, Y_train)}')
    print(f'score:{grid_search.best_score_}')
    print(f'Test accuracy: {resulting_model.score(X_test, Y_test)}')
    train_acc_list = np.append(train_acc_list, resulting_model.score(X_train, Y_train))
    score_list = np.append(score_list, grid_search.best_score_)
    test_acc_list = np.append(test_acc_list, resulting_model.score(X_test, Y_test))

print(f'Test over, results:\n')
print(f'Average train accuracy = {np.mean(train_acc_list)}')
print(f'Average score = {np.mean(score_list)}')
print(f'Average test accuracy = {np.mean(test_acc_list)}')
"""
"""

# Gradient boosting

train_acc_list = np.array([])
score_list = np.array([])
test_acc_list = np.array([])
for i in range(10):
    np.random.seed(i)
    # Shuffle the rows of the dataframe
    df_shuffle = df_robust.sample(frac=1)

    X_test = df_shuffle.iloc[0:200]
    X_train = df_shuffle.drop(X_test.index)
    Y_test = X_test['Lead']
    Y_train = X_train['Lead']
    X_test = X_test.drop(['Lead'], axis=1)
    X_train = X_train.drop(['Lead'], axis=1)

    X_test = X_test.drop(['Number of words lead', 'Total words', 'Gross', 'Year', 'Mean Age Male', 'Mean Age Female', 'Age Lead', 'AgeDiff_1'], axis=1)
    X_train = X_train.drop(['Number of words lead', 'Total words', 'Gross', 'Year', 'Mean Age Male', 'Mean Age Female', 'Age Lead', 'AgeDiff_1'], axis=1)

    # Create a parameter grid to search over
    param_grid = {
        #'loss': []

        'n_estimators': [200],
        'subsample': [0.5],
        'min_samples_split': [30],
        #'min_samples_leaf': [5],
       #'max_features': [None, 'sqrt', 'log2'],


        #'min_impurity_decrease': [0.0, 0.1, 0.2],

       #'init': [None],
       #'validation_fraction': [0.1, 0.2, 0.3],
        #'n_iter_no_change': [None],
        #'warm_start': [True],
    }

    # Create a Gradient boosting classifier object
    GBmodel = GradientBoostingClassifier()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(GBmodel, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, Y_train)

    resulting_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_

    print('\n')
    print(f'Grid search complete for Gradient boosting. Results:\n')
    print(resulting_model)
    print(best_parameters)
    print(f'Train accuracy = {resulting_model.score(X_train, Y_train)}')
    print(f'score:{grid_search.best_score_}')
    print(f'Test accuracy: {resulting_model.score(X_test, Y_test)}')
    train_acc_list = np.append(train_acc_list, resulting_model.score(X_train, Y_train))
    score_list = np.append(score_list, grid_search.best_score_)
    test_acc_list = np.append(test_acc_list, resulting_model.score(X_test, Y_test))

print(f'Test over, results:\n')
print(f'Average train accuracy = {np.mean(train_acc_list)}')
print(f'Average score = {np.mean(score_list)}')
print(f'Average test accuracy = {np.mean(test_acc_list)}')

"""

"""
# AdaBoosting

train_acc_list = np.array([])
score_list = np.array([])
test_acc_list = np.array([])
for i in range(10):
    np.random.seed(i)
    # Shuffle the rows of the dataframe
    df_shuffle = df_robust.sample(frac=1)

    X_test = df_shuffle.iloc[0:200]
    X_train = df_shuffle.drop(X_test.index)
    Y_test = X_test['Lead']
    Y_train = X_train['Lead']
    X_test = X_test.drop(['Lead'], axis=1)
    X_train = X_train.drop(['Lead'], axis=1)

    X_test = X_test.drop(['Age Co-Lead', 'Mean Age Male'], axis=1)
    X_train = X_train.drop(['Age Co-Lead', 'Mean Age Male'], axis=1)

    # Create a parameter grid to search over
    param_grid = {
        'estimator': [b1]
      #'n_estimators': [50, 100, 200],
      # 'learning_rate': [0.1],
        #'algorithm': ['SAMME', 'SAMME.R'],
    }

    # Create a Gradient boosting classifier object
    AdaBoostmodel = AdaBoostClassifier()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(AdaBoostmodel, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, Y_train)

    resulting_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_

    print('\n')
    print(f'Grid search complete for Ada Boosting. Results:\n')
    print(resulting_model)
    print(best_parameters)
    print(f'Train accuracy = {resulting_model.score(X_train, Y_train)}')
    print(f'score:{grid_search.best_score_}')
    print(f'Test accuracy: {resulting_model.score(X_test, Y_test)}')
    train_acc_list = np.append(train_acc_list, resulting_model.score(X_train, Y_train))
    score_list = np.append(score_list, grid_search.best_score_)
    test_acc_list = np.append(test_acc_list, resulting_model.score(X_test, Y_test))

print(f'Test over, results:\n')
print(f'Average train accuracy = {np.mean(train_acc_list)}')
print(f'Average score = {np.mean(score_list)}')
print(f'Average test accuracy = {np.mean(test_acc_list)}')
"""

# create subsets and predict separately for test prediction
# This part was changed to make the final test prediction.
# To go back, change to test_df =df_test, and go back to the top and remove comments 37-45.
test_df = pd.read_csv(r'C:\Users\mariu\PycharmProjects\machineLearning\test.csv')  # read in data in a panda dataframe
test_df['lead_colead_age_diff'] = test_df['Age Lead'] - test_df['Age Co-Lead']

# split in subsets, predict separately with math
single_act = test_df[(test_df['Number of female actors'] == 1) | (test_df['Number of male actors'] == 1)]
test_df = test_df.drop(single_act.index)


# words
words_male = np.array(test_df['Number words male'])
words_female = np.array(test_df['Number words female'])
lead_word = np.array(test_df['Number of words lead'])
word_diff = np.array(test_df['Difference in words lead and co-lead'])
colead_word = lead_word - word_diff

# actors
m_actors = np.array(test_df['Number of male actors'])
f_actors = np.array(test_df['Number of female actors'])
actor_diff = m_actors - f_actors

# age
lead_age = np.array(test_df['Age Lead'])
co_lead_age = np.array(test_df['Age Co-Lead'])
age_difference = lead_age - co_lead_age

# female words
female_colead_diff = words_female - colead_word  # if negative lead = female
male_colead_diff = words_male - colead_word  # if negative lead = male or maybe <100
avg_male_word = male_colead_diff / (m_actors - 1)
avg_female_word = female_colead_diff / (f_actors - 1)

# criterion for splits
mask2 = (avg_female_word < 100) | (avg_female_word > colead_word)  # Lead female
mask3 = (avg_male_word < 100) | (avg_male_word > colead_word)  # Lead male

# subsets
subset1 = single_act
subset2 = test_df[mask2]
subset3 = test_df[mask3]

# drop subsets (1 is already dropped)
test_df = test_df.drop(subset2.index)
test_df = test_df.drop(subset3.index)

# age diff
m_actors = np.array(test_df['Number of male actors'])
f_actors = np.array(test_df['Number of female actors'])
actor_diff = m_actors - f_actors
lead_age = np.array(test_df['Age Lead'])
co_lead_age = np.array(test_df['Age Co-Lead'])
age_difference = lead_age - co_lead_age

# last criterion
mask4 = (actor_diff < 0) & (
        age_difference < -1)  # Lead probably a female, 0.97%. Might be overfitting. Try with and without it. create a dummy instead?
subset4 = test_df[mask4]
test_df = test_df.drop(subset4.index)

# prediction 1
words_male = np.array(subset1['Number words male'])
words_female = np.array(subset1['Number words female'])
number_men = np.array(subset1['Number of male actors'])
predictions1 = np.where((words_female == 0) | ((words_male != 0) & (number_men == 1)), 'Female', 'Male')

# prediction 2, 3, 4
predictions2 = np.full(subset2.shape[0], 'Female')
predictions3 = np.full(subset3.shape[0], 'Male')
predictions4 = np.full(subset4.shape[0], 'Female')

# count
removed_count = predictions3.shape[0] + predictions2.shape[0] + predictions1.shape[0] + predictions4.shape[0]
print(
    f'{predictions1.shape[0]} + {predictions2.shape[0]} + {predictions3.shape[0]} + {predictions4.shape[0]} = {removed_count}')

print(f'{test_df.shape[0]} + {removed_count} = {removed_count + test_df.shape[0]}')
print('total: ')
print(removed_count + test_df.shape[0])

# This part was used before the final test predictions
"""
# subset 1 prediction
print(f'subset 1:')
y_test = np.array(subset1['Lead'])
accuracy1 = np.mean(predictions1 == y_test)
print(f'accuracy: {accuracy1} ')
conf_matrix = pd.crosstab(predictions1, y_test)
print(f'{conf_matrix}\n')

# subset 2 prediction
print(f'subset 2:')
y_test = np.array(subset2['Lead'])
accuracy2 = np.mean(predictions2 == y_test)
print(f'accuracy: {accuracy2} ')
conf_matrix = pd.crosstab(predictions2, y_test)
print(f'{conf_matrix}\n')

# subset 3 prediction
print(f'subset 3:')
y_test = np.array(subset3['Lead'])
accuracy3 = np.mean(predictions3 == y_test)
print(f'accuracy: {accuracy3} ')
conf_matrix = pd.crosstab(predictions3, y_test)
print(f'{conf_matrix}\n')

# subset 4 prediction
print(f'subset 4:')
y_test = np.array(subset4['Lead'])
accuracy4 = np.mean(predictions4 == y_test)
print(f'accuracy: {accuracy4} ')
conf_matrix = pd.crosstab(predictions4, y_test)
print(f'{conf_matrix}\n')

"""

# Model prediction
print(f'main set prediction:')
model = RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_split=15)
df = df.drop(
    ['Mean Age Male', 'Age Lead', 'Age Co-Lead',
     'Difference in words lead and co-lead', 'Number words male',
     'Number of words lead', 'Total words', 'Mean Age Female',
     'Number words female'], axis=1)

test_df = test_df.drop(['Mean Age Male', 'Age Lead', 'Age Co-Lead',
                        'Difference in words lead and co-lead', 'Number words male',
                        'Number of words lead', 'Total words', 'Mean Age Female',
                        'Number words female'], axis=1)


data_frame = df
x_test = test_df
x_train = df
#y_test = x_test['Lead']
y_train = x_train['Lead']
#x_test = x_test.drop(['Lead'], axis=1)
x_train = x_train.drop(['Lead'], axis=1)
# Use x_train and x_test as the training and validation sets, respectively

# naive accuracy
"""
z = np.zeros(y_test.shape[0])
naive_p = np.where(z == 0, 'Male', 'Male')
naive_a = np.mean(naive_p == y_test)
print(naive_a)
"""

# model fit
model.fit(x_train, y_train)

# prediction, probability
prediction_probability = model.predict_proba(x_test)
prediction = np.empty(len(x_test), dtype=object)
prediction = np.where(prediction_probability[:, 0] > 0.5, 'Female', 'Male')


# Accuracy and confusion matrix
"""
accuracy = np.mean(prediction == y_test)
conf_matrix = pd.crosstab(prediction, y_test)
print(f'accuracy: {accuracy} ')
print(f'score: {model.score(x_test, y_test)}')
print(f'Naive accuracy: {naive_a}')
print(f'confusion matrix:\n{conf_matrix}')
"""

# add predictions
subset1['prediction'] = predictions1
subset2['prediction'] = predictions2
subset3['prediction'] = predictions3
subset4['prediction'] = predictions4
test_df['prediction'] = prediction

# keep prediction and class
"""
subset1 = subset1[['Lead', 'prediction']]
subset2 = subset2[['Lead', 'prediction']]
subset3 = subset3[['Lead', 'prediction']]
subset4 = subset4[['Lead', 'prediction']]
test_df = test_df[['Lead', 'prediction']]
"""

# combine predictions
combined_df = pd.concat([subset1, subset2, subset3, subset4, test_df], axis=0)
combined_df = combined_df.sort_index()
combined_df = combined_df.drop(['Mean Age Male', 'Age Lead', 'Age Co-Lead',
                        'Difference in words lead and co-lead', 'Number words male',
                        'Number of words lead', 'Total words', 'Mean Age Female',
                        'Number words female'], axis=1)


# Final prediction
prediction = np.array(combined_df['prediction'])

# print results
"""
#y_test = np.array(combined_subset['Lead'])
y_test = y_safety
# Accuracy and confusion matrix
accuracy = np.mean(prediction == y_test)
conf_matrix = pd.crosstab(prediction, y_test)
print(f'accuracy: {accuracy} ')
print(f'Naive accuracy: {naive_a}')
print(f'confusion matrix:\n{conf_matrix}')
"""

# Save predictions in csv file
prediction = np.where(prediction == 'Female', 1, 0)

test_prediction_list = prediction.tolist()
print(test_prediction_list)

"""
with open('predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(test_prediction_list)
"""

