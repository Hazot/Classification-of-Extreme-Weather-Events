from solution import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(3395)

# Raw data
train_data = pd.read_csv("train.csv").iloc[:, 1:]
test_data = pd.read_csv("test.csv").iloc[:, 1:]
train_data.sort_values(by=['LABELS'], axis=0)

# Separate data
train_inputs = train_data.iloc[:, :-1]
train_labels = train_data.iloc[:, -1:]
test_inputs = test_data.iloc[:, :-1]
test_labels = test_data.iloc[:, -1:]

# Lightened the training data
train_sorted = train_data.sort_values(by=['LABELS'], axis=0, ascending=False)
train_inputs_light = train_sorted.iloc[5000:13000, :-1]
train_labels_opt = train_sorted.iloc[5000:13000, -1:]

# df_norm = (df - df.mean()) / (df.max() - df.min())
# df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

# Count the number of 0, 1 and 2
# a = train_labels_opt.to_numpy()
# print(np.sum(a == 0))
# print(np.sum(a == 1))
# print(np.sum(a == 2))

# Center
train_inputs_opt = train_inputs_light.subtract(train_inputs_light.mean())

# log_model = LogisticRegression()
#
# log_model.train(train_inputs_opt, train_labels_opt)
#
# log_model.gradient_descent(1, 0.001, 100)

# w_test = np.random.random((len(train_inputs_opt[1, :]), 3))
# w_test = np.zeros((train_inputs_opt.shape[1], 3))
# w_star_test, b_star_test = gradient_descent(train_inputs_opt, train_labels_opt, w_test, 0, 0.01, 0.1, 100)
# print(w_star_test, b_star_test)
