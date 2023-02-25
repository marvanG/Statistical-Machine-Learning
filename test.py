import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 15)  # see up to 15 columns
np.set_printoptions(suppress=True)  # shows 0.07 instead of 7*e^-2

# Data preparation
data_frame = pd.read_csv(
    r'C:\Users\mariu\PycharmProjects\machineLearning\train.csv')  # read in data in a panda dataframe
Y = data_frame['Lead']
# print(Y)
data_frame = data_frame.drop(['Lead'], axis=1)

# Convert dataframe to numpy array
df_values = data_frame.values
# print(np.array(data_frame.columns))
# Normalize the numpy array
norm_df_values = skl_pre.normalize(df_values)
# Create a new dataframe from the normalized array
norm_data_frame = pd.DataFrame(norm_df_values, columns=data_frame.columns)
norm_data_frame['Lead'] = Y

# General information
# print(data_frame)
# print(data_frame.info())
# print(data_frame.describe())


lista = np.array(data_frame.columns)

df = data_frame
iteration = 0
global_max_accuracy = 0
max_accuracy_cols = []
while len(lista) > 1:
    iteration += 1
    local_max_accuracy = 0
    for i in range(len(lista)):
        accuracy = random.random()
        if accuracy > local_max_accuracy:
            local_max_accuracy = accuracy
            col_to_drop = i
        elif accuracy == local_max_accuracy:
            print(f'Accuracy == for {lista[i]}')

    print(f'Iteration: {iteration}')
    print(f'Column to drop is: {lista[col_to_drop]}')
    df = df.drop(lista[col_to_drop], axis=1)
    lista = np.array(df.columns)
    print(f'Columns left: \n{lista}')
    print(f'Accuracy = {local_max_accuracy} \n')

    if local_max_accuracy > global_max_accuracy:
        global_max_accuracy = local_max_accuracy
        max_accuracy_cols = lista

print('\nFinished\n')
print(f'Best accuracy = {global_max_accuracy} with {len(max_accuracy_cols)} features: \n{max_accuracy_cols}')