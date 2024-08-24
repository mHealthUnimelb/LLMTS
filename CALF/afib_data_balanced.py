"""
    Preprocessing module for MIT_BIH waveform database
    -------
    This module provides classes and methods for creating the MIT-BIH Atrial Fibrillation database.
    Original source: https://github.com/Seb-Good/deepecg
    """

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import wfdb
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.utils import shuffle
import scipy
from sklearn.model_selection import train_test_split

# Local imports
# from deepecg.config.config import DATA_DIR
DATA_DIR = "./datasets"
afib_dict = {"AFIB": 0, "AFL": 1, "J": 2, "N": 3}


class AFDB(object):
    """
        The MIT-BIH Atrial Fibrillation Database
        https://physionet.org/physiobank/database/afdb/
        """

    def __init__(self):
        # Set attributes
        self.db_name = 'afdb'
        self.raw_path = os.path.join(DATA_DIR, 'ECG_raw')
        self.processed_path = os.path.join(DATA_DIR, 'processed_balanced')
        self.label_dict = {'AFIB': 'atrial fibrillation', 'AFL': 'atrial flutter', 'J': 'AV junctional rhythm'}
        self.fs = 300
        self.length = 60
        self.length_sp = self.length * self.fs
        self.record_ids = None
        self.sections = None
        self.samples = None
        self.labels = None

    def generate_db(self):
        """Generate raw and processed databases."""
        # Generate raw database
        self.generate_raw_db()

        # Generate processed database
        self.generate_processed_db()

    def generate_raw_db(self):
        """Generate the raw version of the MIT-BIH Atrial Fibrillation database in the 'raw' folder."""
        # Download database
        if len(os.listdir(self.raw_path)) == 0:
            print('Generating Raw MIT-BIH Atrial Fibrillation Database ...')
            wfdb.dl_database(self.db_name, self.raw_path)
            print('Complete!\n')

        # Get list of recordings
        self.record_ids = [file.split('.')[0] for file in os.listdir(self.raw_path) if '.dat' in file]

    def check_distribution(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

    def plot_signals(self, signals, num_plots=5):
        plt.figure(figsize=(15, 8))

        for i in range(min(num_plots, signals.shape[0])):
            plt.subplot(num_plots, 1, i+1)
            plt.plot(signals[i][0]) # plot first channel
            plt.title(f"Segment {i + 1}")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()


    def generate_processed_db(self):
        """Generate the processed version of the MIT-BIH Atrial Fibrillation database in the 'processed' folder."""
        print('Generating Processed MIT-BIH Atrial Fibrillation Database ...')
        all_signals, all_labels = self._get_sections()

        signal_lens = [len(sig) for sig in all_labels]
        print("signal_lens: ", len(signal_lens))
        all_signals = np.array([sig[:,:min(signal_lens)] for sig in all_signals])
        all_labels = np.array([sig[:min(signal_lens)] for sig in all_labels])
        #
        # n_train = int(0.6*len(all_signals))
        # n_val = int(0.8*len(all_signals))
        # train_data = all_signals[:n_train]
        # val_data = all_signals[n_train:n_val]
        # test_data = all_signals[n_val:]
        # train_state = all_labels[:n_train]
        # val_state = all_labels[n_train:n_val]
        # test_state = all_labels[n_val:]

        # create balanced split
        # train_data, train_state, val_data, val_state, test_data, test_state = self._create_balanced_split(data)

        # segment data
        segmented_signals, segmented_labels = self._segment_data(all_signals, all_labels, 2500, strategy='discard')

        # Plot segmented signals
        # self.plot_signals(segmented_signals)

        # create balanced split
        # train_data, train_state, val_data, val_state, test_data, test_state = self._create_balanced_split(segmented_signals)

        # split data into train, val and test set while ensuring each class is balanced
        x_train, x_temp, y_train, y_temp = train_test_split(segmented_signals, segmented_labels, test_size=0.4, random_state=42, stratify=segmented_labels)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        # Normalize signals
        # train_data_n, val_data_n, test_data_n = self._normalize(train_data, val_data, test_data)
        x_train, x_val, x_test = self._normalize(x_train, x_val, x_test)

        # Plot segmented signals
        # self.plot_signals(x_test)

        # check if each class is balanced
        train_distribution = self.check_distribution(y_train)
        validation_distribution = self.check_distribution(y_val)
        test_distribution = self.check_distribution(y_test)

        print("Train Set Class Distribution:", train_distribution)
        print("Validation Set Class Distribution:", validation_distribution)
        print("Test Set Class Distribution:", test_distribution)

        # Save signals to file
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)
        with open(os.path.join(self.processed_path, 'x_train.pkl'), 'wb') as f:
            pickle.dump(x_train, f)
        with open(os.path.join(self.processed_path, 'x_val.pkl'), 'wb') as f:
            pickle.dump(x_val, f)
        with open(os.path.join(self.processed_path, 'x_test.pkl'), 'wb') as f:
            pickle.dump(x_test, f)
        with open(os.path.join(self.processed_path, 'state_train.pkl'), 'wb') as f:
            pickle.dump(y_train, f)
        with open(os.path.join(self.processed_path, 'state_val.pkl'), 'wb') as f:
            pickle.dump(y_val, f)
        with open(os.path.join(self.processed_path, 'state_test.pkl'), 'wb') as f:
            pickle.dump(y_test, f)

    def _segment_data(self, all_signals, all_labels, seq_len=2500, strategy='discard'):
        num_samples, num_channels, total_length = all_signals.shape

        if strategy == "discard":
            # discard the last segment if it is shorter than seq_len
            num_segments = total_length // seq_len
            x_data = all_signals[:, :, :num_segments * seq_len]
            y_data = all_labels[:, :num_segments * seq_len]
        elif strategy == "pad":
            # pad the last segment if it is shorter than seq_len
            num_segments = np.ceil(total_length / seq_len).astype(int)
            x_data = np.pad(all_signals, ((0, 0), (0, 0), (0, num_segments * seq_len - total_length)), 'constant', constant_values=0)
            y_data = np.pad(all_labels, ((0, 0), (0, num_segments * seq_len - total_length)), 'constant', constant_values=0)

        # reshape x_data to (num_samples * num_segments, seq_len, num_channels)
        # x_data = x_data.reshape(num_samples, num_channels, num_segments, seq_len).transpose(0, 2, 3, 1).reshape(num_samples * num_segments, seq_len, num_channels)

        # reshape x_data to (num_samples * num_segments, num_channels, seq_len)
        x_data = x_data.reshape(num_samples, num_channels, num_segments, seq_len).transpose(0, 2, 1, 3).reshape(num_samples * num_segments, num_channels, seq_len)

        # reshape y_data to (num_samples * num_segments, )
        y_data = y_data.reshape(num_samples, num_segments, seq_len)
        # compute the mode for each segment, find the most common label
        mode_labels = scipy.stats.mode(y_data, axis=2).mode
        y_data = mode_labels.reshape(num_samples * num_segments)

        # combined_data = np.stack([x_data, y_data], axis=1)
        return x_data, y_data


    # def _create_balanced_split(self, data):
    #     # balancing classes across splits
    #     labels = [lab for sig, lab in data]
    #     unique_labels = np.unique(labels)
    #     print("unique_labels: ", unique_labels)
    #     label_to_data = {label: [] for label in unique_labels}
    #
    #     for sig, label in data:
    #         label_to_data[label].append(sig)
    #
    #     min_samples_per_class = min(len(label_to_data[label]) for label in unique_labels)
    #     n_train = int(0.6 * min_samples_per_class)
    #     n_val = int(0.8 * min_samples_per_class)
    #
    #     train_signals, train_labels = [], []
    #     val_signals, val_labels = [], []
    #     test_signals, test_labels = [], []
    #
    #     for label in unique_labels:
    #         signals = label_to_data[label]
    #         train_signals.extend(signals[:n_train])
    #         train_labels.extend([label] * n_train)
    #         val_signals.extend(signals[n_train:n_val])
    #         val_labels.extend([label] * (n_val - n_train))
    #         test_signals.extend(signals[n_val:])
    #         test_labels.extend([label] * (len(signals) - n_val))
    #
    #     train_data = np.array(train_signals)
    #     train_state = np.array(train_labels)
    #     val_data = np.array(val_signals)
    #     val_state = np.array(val_labels)
    #     test_data = np.array(test_signals)
    #     test_state = np.array(test_labels)
    #
    #     return train_data, train_state, val_data, val_state, test_data, test_state

    def _normalize(self, train_data, val_data, test_data):
        """ Calculate the mean and std of each feature from the training set
        """
        feature_means = np.mean(train_data, axis=(0, 2))
        feature_std = np.std(train_data, axis=(0, 2))
        train_data_n = (train_data - feature_means[np.newaxis, :, np.newaxis]) / \
                       np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
        val_data_n = (val_data - feature_means[np.newaxis, :, np.newaxis]) / \
                     np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
        test_data_n = (test_data - feature_means[np.newaxis, :, np.newaxis]) / \
                      np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
        return train_data_n, val_data_n, test_data_n

    def _get_sections(self):
        """Collect continuous arrhythmia sections."""
        # Empty dictionary for arrhythmia sections
        all_signals = []
        all_labels = []

        # Loop through records
        for record_id in self.record_ids:
            # Import recording
            record = wfdb.rdrecord(os.path.join(self.raw_path, record_id))

            # Import annotations
            annotation = wfdb.rdann(os.path.join(self.raw_path, record_id), 'atr')

            # Get sample frequency
            fs = record.__dict__['fs']

            # Get waveform
            waveform = record.__dict__['p_signal']  # shape: (length, n_channels=2)

            # labels
            labels = [label[1:] for label in annotation.__dict__['aux_note']]

            # Samples
            sample = annotation.__dict__['sample']

            padded_labels = np.zeros(len(waveform))
            for i, l in enumerate(labels):
                if i == len(labels) - 1:
                    padded_labels[sample[i]:] = afib_dict[l]
                else:
                    padded_labels[sample[i]:sample[i + 1]] = afib_dict[l]
            padded_labels = padded_labels[sample[0]:]
            all_labels.append(padded_labels)
            all_signals.append(waveform[sample[0]:, :].T)

        return all_signals, all_labels


if __name__ == "__main__":
    a = AFDB()
    a.generate_raw_db()
    a.generate_processed_db()
