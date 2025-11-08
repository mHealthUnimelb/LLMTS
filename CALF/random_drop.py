import pandas as pd
import numpy as np
import os
import pickle
import argparse
from matplotlib import pyplot as plt


def random_drop_data(input_pkl_file, drop_percentage, action='drop', output_pkl_file='datasets/ECG/x_test_dropped.pkl'):
    with open(input_pkl_file, 'rb') as file:
        data = pd.read_pickle(file)

    # randomly drop data from the last dimension (seq_len)
    num_time_points = data.shape[2]
    num_modified = int(num_time_points * drop_percentage)

    # randomly select the percentage of time points
    indices = np.random.choice(num_time_points, num_modified, replace=False)

    if action == 'drop':
        # drop the selected time points from the last dimension
        data_modified = np.delete(data, indices, axis=2)
    elif action == 'zero':
        # set the selected time points to zero
        data_modified = data
        data_modified[:, :, indices] = 0
    else:
        raise ValueError('Invalid action. Choose either "drop" or "zero"')

    # save dropped data to file
    # file_path = os.path.dirname(output_pkl_file)
    # if not os.path.exists(file_path):
    #     os.mkdir(file_path)
    with open(output_pkl_file, 'wb') as f:
        pickle.dump(data_modified, f)

    return data_modified


def plot_signals(signals, num_plots=5):
    plt.figure(figsize=(15, 8))

    for i in range(min(num_plots, signals.shape[0])):
        plt.subplot(num_plots, 1, i+1)
        plt.plot(signals[i][0]) # plot first channel
        plt.title(f"Segment {i + 1}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Randomly drop or zero data in a pickle file.")
    parser.add_argument('input_pkl_file', type=str, help="Path to the input pickle file.")
    parser.add_argument('drop_percentage', type=float, help="Percentage of time points to drop.")
    parser.add_argument('action', type=str, choices=['drop', 'zero'], help="Action to perform: 'drop' or 'zero'.")
    parser.add_argument('output_pkl_file', type=str, help="Path to save the output pickle file.")

    args = parser.parse_args()

    # Call the function with command-line arguments
    random_drop_data(args.input_pkl_file, args.drop_percentage, args.action, args.output_pkl_file)

    with open(args.input_pkl_file, 'rb') as file:
        data1 = pd.read_pickle(file)
    print("Original shape: ", data1.shape)

    with open(args.output_pkl_file, 'rb') as file:
        data2 = pd.read_pickle(file)
    print(f"Shape after dropping {args.drop_percentage} of data: {data2.shape}")

if __name__ == '__main__':
    main()
