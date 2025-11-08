# train/val/test

import torch
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt
import argparse
import math
import seaborn as sns; sns.set()
import sys
from sklearn.model_selection import train_test_split

import numpy as np
import pickle
import os
import random
random.seed(0)

def check_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

# The waveform dataset is very small and sparse. If due to class imbalance there are no samples of a
#  particular class in the test set, use the training set onlly for evaluation.

data_path = './datasets/waveform_data/processed'
with open(os.path.join(data_path, 'x_train.pkl'), 'rb') as f:
    x = pickle.load(f)
    T = x.shape[-1]
    x_window = np.concatenate(np.split(x[:, :, :T // 5 * 5], 5, -1), 0)

with open(os.path.join(data_path, 'state_train.pkl'), 'rb') as f:
        y = pickle.load(f)
        y_window = np.concatenate(np.split(y[:, :T // 5 * 5], 5, -1), 0)

print(x_window.shape, y_window.shape)

x = x_window
y = y_window
window_size = 2500
T=x.shape[-1]
x_window = np.split(x[:, :, :window_size * (T // window_size)], (T // window_size), -1)
x_window = np.concatenate(x_window, 0)

y_window = np.concatenate(np.split(y[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
y_window = np.array([np.bincount(yy).argmax() for yy in y_window])

print(x_window.shape, y_window.shape)

# then split the data into train, valid, test 6:2:2 as your experiment setting
# split data into train, val and test set while ensuring each class is balanced
x_train, x_temp, y_train, y_temp = train_test_split(x_window, y_window, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# check if each class is balanced
train_distribution = check_distribution(y_train)
validation_distribution = check_distribution(y_val)
test_distribution = check_distribution(y_test)

print("Train Set Class Distribution:", train_distribution)
print("Validation Set Class Distribution:", validation_distribution)
print("Test Set Class Distribution:", test_distribution)

print(f"train shape: {x_train.shape} type: {type(x_train)}")
print(f"val shape: {x_val.shape} type: {type(x_val)}")
print(f"test shape: {x_test.shape} type: {type(x_test)}")

# Save signals to file
processed_path = './datasets/ECG'
if not os.path.exists(processed_path):
    os.mkdir(processed_path)
with open(os.path.join(processed_path, 'x_train.pkl'), 'wb') as f:
    pickle.dump(x_train, f)
with open(os.path.join(processed_path, 'x_val.pkl'), 'wb') as f:
    pickle.dump(x_val, f)
with open(os.path.join(processed_path, 'x_test.pkl'), 'wb') as f:
    pickle.dump(x_test, f)
with open(os.path.join(processed_path, 'state_train.pkl'), 'wb') as f:
    pickle.dump(y_train, f)
with open(os.path.join(processed_path, 'state_val.pkl'), 'wb') as f:
    pickle.dump(y_val, f)
with open(os.path.join(processed_path, 'state_test.pkl'), 'wb') as f:
    pickle.dump(y_test, f)