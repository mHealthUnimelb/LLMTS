import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.utils import shuffle
import scipy
from sklearn.model_selection import train_test_split


# def check_distribution(labels):
#     unique, counts = np.unique(labels, return_counts=True)
#     return dict(zip(unique, counts))
#
#
# def plot_signals(signals, num_plots=5):
#     plt.figure(figsize=(15, 8))
#
#     for i in range(min(num_plots, signals.shape[0])):
#         plt.subplot(num_plots, 1, i + 1)
#         plt.plot(signals[i][0])  # plot first channel
#         plt.title(f"Segment {i + 1}")
#         plt.xlabel("Time")
#         plt.ylabel("Amplitude")
#
#     plt.tight_layout()
#     plt.show()
#
#
# def segment_data(all_signals, all_labels, seq_len=2500, strategy='discard'):
#     num_samples, num_channels, total_length = all_signals.shape
#
#     if strategy == "discard":
#         # discard the last segment if it is shorter than seq_len
#         num_segments = total_length // seq_len
#         x_data = all_signals[:, :, :num_segments * seq_len]
#         y_data = all_labels[:, :num_segments * seq_len]
#     elif strategy == "pad":
#         # pad the last segment if it is shorter than seq_len
#         num_segments = np.ceil(total_length / seq_len).astype(int)
#         x_data = np.pad(all_signals, ((0, 0), (0, 0), (0, num_segments * seq_len - total_length)), 'constant',
#                         constant_values=0)
#         y_data = np.pad(all_labels, ((0, 0), (0, num_segments * seq_len - total_length)), 'constant', constant_values=0)
#
#     # reshape x_data to (num_samples * num_segments, seq_len, num_channels)
#     # x_data = x_data.reshape(num_samples, num_channels, num_segments, seq_len).transpose(0, 2, 3, 1).reshape(num_samples * num_segments, seq_len, num_channels)
#
#     # reshape x_data to (num_samples * num_segments, num_channels, seq_len)
#     x_data = x_data.reshape(num_samples, num_channels, num_segments, seq_len).transpose(0, 2, 1, 3).reshape(
#         num_samples * num_segments, num_channels, seq_len)
#
#     # reshape y_data to (num_samples * num_segments, )
#     y_data = y_data.reshape(num_samples, num_segments, seq_len)
#     # compute the mode for each segment, find the most common label
#     mode_labels = scipy.stats.mode(y_data, axis=2).mode
#     y_data = mode_labels.reshape(num_samples * num_segments)
#
#     # combined_data = np.stack([x_data, y_data], axis=1)
#     return x_data, y_data
#
#
# with open('datasets/waveform_data/processed/x_train.pkl', 'rb') as file:
#     train_data = pickle.load(file)
#
# with open('datasets/waveform_data/processed/state_train.pkl', 'rb') as file:
#     train_state = pickle.load(file)
#
# print("train shape: ", train_data.shape)
# print("state shape: ", train_state.shape)
#
# segmented_signals, segmented_labels = segment_data(train_data, train_state, 2500, strategy='discard')
#
# print("segmented_signals shape: ", segmented_signals.shape)
# print("segmented_labels shape: ", segmented_labels.shape)
#
# # Plot segmented signals
# # self.plot_signals(segmented_signals)
#
# # split data into train, val and test set while ensuring each class is balanced
# x_train, x_temp, y_train, y_temp = train_test_split(segmented_signals, segmented_labels, test_size=0.4, random_state=42,
#                                                     stratify=segmented_labels)
# x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
#
# # Plot segmented signals
# # self.plot_signals(x_test)
#
# # check if each class is balanced
# train_distribution = check_distribution(y_train)
# validation_distribution = check_distribution(y_val)
# test_distribution = check_distribution(y_test)
#
# print("Train Set Class Distribution:", train_distribution)
# print("Validation Set Class Distribution:", validation_distribution)
# print("Test Set Class Distribution:", test_distribution)
#
# print("train shape: ", x_train.shape)
# print("val shape: ", x_val.shape)
# print("test shape: ", x_test.shape)

#
# # Save signals to file
# processed_path = './datasets/ECG'
# if not os.path.exists(processed_path):
#     os.mkdir(processed_path)
# with open(os.path.join(processed_path, 'x_train.pkl'), 'wb') as f:
#     pickle.dump(x_train, f)
# with open(os.path.join(processed_path, 'x_val.pkl'), 'wb') as f:
#     pickle.dump(x_val, f)
# with open(os.path.join(processed_path, 'x_test.pkl'), 'wb') as f:
#     pickle.dump(x_test, f)
# with open(os.path.join(processed_path, 'state_train.pkl'), 'wb') as f:
#     pickle.dump(y_train, f)
# with open(os.path.join(processed_path, 'state_val.pkl'), 'wb') as f:
#     pickle.dump(y_val, f)
# with open(os.path.join(processed_path, 'state_test.pkl'), 'wb') as f:
#     pickle.dump(y_test, f)



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