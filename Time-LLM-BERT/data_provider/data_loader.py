import os
import numpy as np
import pandas as pd
import scipy
from sklearn import model_selection
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.physionet import PhysioNet, get_data_min_max, variable_time_collate_fn2
import data_provider.utils
import warnings
import torch

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_ECG(Dataset):

    def __init__(self, root_path, data_path=None, flag='train', seq_len=2500, label_len=1, pred_len=1, scale=True, percent=100, training_flag=1):
        if training_flag == 1:
            # load data
            if flag == "train":
                self.x_data = pd.read_pickle(root_path + "x_train.pkl")
                self.y_data = pd.read_pickle(root_path + "state_train.pkl")
            elif flag == "val":
                self.x_data = pd.read_pickle(root_path + "x_val.pkl")
                self.y_data = pd.read_pickle(root_path + "state_val.pkl")
            elif flag == "test":
                self.x_data = pd.read_pickle(root_path + "x_test.pkl")
                self.y_data = pd.read_pickle(root_path + "state_test.pkl")
        else:
            self.x_data = pd.read_pickle(root_path + data_path)
            self.y_data = pd.read_pickle(root_path + "state_test.pkl")
            print("data_path: ", data_path)
        self.class_names = ['AFIB', 'AFL', 'J', 'N']
        self.max_seq_len = seq_len
        self.feature_dim = self.x_data.shape[1]

        self.x_data = self.x_data.transpose(0, 2, 1)
        # self.x_data = torch.from_numpy(self.x_data)
        # self.y_data = torch.tensor(self.y_data, dtype=torch.long)

    #     # segment data
    #     self.segment_data(seq_len, strategy="discard")
    #
    # def segment_data(self, seq_len, strategy="discard"):
    #     num_samples, num_channels, total_length = self.x_data.shape
    #
    #     if strategy == "discard":
    #         # discard the last segment if it is shorter than seq_len
    #         num_segments = total_length // seq_len
    #         self.x_data = self.x_data[:, :, :num_segments * seq_len]
    #         self.y_data = self.y_data[:, :num_segments * seq_len]
    #     elif strategy == "pad":
    #         # pad the last segment if it is shorter than seq_len
    #         num_segments = np.ceil(total_length / seq_len).astype(int)
    #         self.x_data = np.pad(self.x_data, ((0, 0), (0, 0), (0, num_segments * seq_len - total_length)),
    #                              mode='constant', constant_values=0)
    #         self.y_data = np.pad(self.y_data, ((0, 0), (0, num_segments * seq_len - total_length)), mode='constant',
    #                              constant_values=0)
    #
    #     # reshape x_data to (num_samples * num_segments, num_channels, seq_len)
    #     self.x_data = self.x_data.reshape(num_samples, num_channels, num_segments, seq_len).transpose(0, 2, 3,
    #                                                                                                   1).reshape(
    #         num_samples * num_segments, seq_len, num_channels)
    #
    #     # reshape y_data to (num_samples * num_segments, 1)
    #     self.y_data = self.y_data.reshape(num_samples, num_segments, seq_len)
    #     # compute the mode for each segment, find the most common label
    #     # mode_labels = np.argmax(np.sum(self.y_data, axis=2), axis=1)
    #     mode_labels = scipy.stats.mode(self.y_data, axis=2).mode
    #     self.y_data = mode_labels.reshape(num_samples * num_segments, 1)
    #
    #     # reshape y_data to (num_samples * num_segments, 1)
    #     # self.y_data = self.y_data.reshape((-1, 1))
    #     # reshape y_data to (num_samples * num_segments, seq_len)
    #     # self.y_data = self.y_data.reshape(num_samples, num_segments, seq_len).reshape(num_samples * num_segments,
    #     #                                                                               seq_len)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


class Dataset_Physionet(Dataset):
    def __init__(self, root_path, data_path=None, args=None, device='cpu', dataset_flag=1, flag='train', q=0, seq_len=2500, training_flag=1):
        # train_dataset_obj = PhysioNet('data/physionet', train=True,
        #                               quantization=q,
        #                               download=True, n_samples=min(10000, args.n),
        #                               device=device)
        # # Use custom collate_fn to combine samples with arbitrary time observations.
        # # Returns the dataset along with mask and time steps
        # test_dataset_obj = PhysioNet('data/physionet', train=False,
        #                              quantization=q,
        #                              download=True, n_samples=min(10000, args.n),
        #                              device=device)

        total_dataset = PhysioNet('data/physionet',
                                  quantization=q,
                                  download=True,
                                  device=device)

        # Combine and shuffle samples from physionet Train and physionet Test
        # total_dataset = train_dataset_obj[:len(train_dataset_obj)]

        # if not args.classif:
        #     # Concatenate samples from original Train and Test sets
        #     # Only 'training' physionet samples are have labels.
        #     # Therefore, if we do classifiction task, we don't need physionet 'test' samples.
        #     total_dataset = total_dataset + \
        #                     test_dataset_obj[:len(test_dataset_obj)]
        print("total_dataset shape:", len(total_dataset))
        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                                 random_state=42, shuffle=True)
        if args.classif:
            # if classification task, we further split into train and validation sets
            val_data, test_data = model_selection.train_test_split(test_data, train_size=0.5,
                                                                   random_state=42, shuffle=False)

        record_id, tt, vals, mask, labels = train_data[0]

        # n_samples = len(total_dataset)
        input_dim = vals.size(-1)
        data_min, data_max = get_data_min_max(total_dataset, device)
        batch_size = min(len(train_data), args.batch_size)
        if dataset_flag:
            test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif,
                                                          data_min=data_min, data_max=data_max)

            if args.classif:
                # train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                #                                                         random_state=11, shuffle=True)
                train_data_combined = variable_time_collate_fn(
                    train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
                val_data_combined = variable_time_collate_fn(
                    val_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
                print(train_data_combined[1].sum(
                ), val_data_combined[1].sum(), test_data_combined[1].sum())
                print(train_data_combined[0].size(), train_data_combined[1].size(),
                      val_data_combined[0].size(), val_data_combined[1].size(),
                      test_data_combined[0].size(), test_data_combined[1].size())
                self.time_steps = train_data_combined[0].size()[1]

                train_data_combined = TensorDataset(
                    train_data_combined[0], train_data_combined[1].long().squeeze())
                val_data_combined = TensorDataset(
                    val_data_combined[0], val_data_combined[1].long().squeeze())
                test_data_combined = TensorDataset(
                    test_data_combined[0], test_data_combined[1].long().squeeze())
            else:
                train_data_combined = variable_time_collate_fn(
                    train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
                print(train_data_combined.size(), test_data_combined.size())

            train_dataloader = DataLoader(
                train_data_combined, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(
                test_data_combined, batch_size=batch_size, shuffle=False)

        else:
            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                          collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device,
                                                                                             data_type="train",
                                                                                             data_min=data_min,
                                                                                             data_max=data_max))
            test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                         collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device,
                                                                                            data_type="test",
                                                                                            data_min=data_min,
                                                                                            data_max=data_max))

        attr_names = total_dataset.params
        data_objects = {"dataset_obj": total_dataset,
                        "train_dataloader": train_dataloader,
                        "test_dataloader": test_dataloader,
                        "input_dim": input_dim,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader),
                        "attr": attr_names,  # optional
                        "classif_per_tp": False,  # optional
                        "n_labels": 1}  # optional
        if args.classif:
            val_dataloader = DataLoader(
                val_data_combined, batch_size=batch_size, shuffle=False)
            data_objects["val_dataloader"] = val_dataloader
        self.data_objects = data_objects
        self.class_names = ["survival", "death"]
        print("time steps: ", self.time_steps)

class Dataset_P12(Dataset):
    def __init__(self, args=None, root_path=None, dataset='P12', device=torch.device("cpu"), q=0.016, upsampling_batch=False):
        self.data_objects = data_provider.utils.get_data(args, dataset, device, q, upsampling_batch)
        self.class_names = ["survival", "death"]

class Dataset_MIMIC(Dataset):
    def __init__(self, args=None, root_path=None, dataset='MIMIC', device=torch.device("cpu"), q=0.016, upsampling_batch=False):
        self.data_objects = data_provider.utils.get_data(args, dataset, device, q, upsampling_batch)
        self.class_names = [0, 1]

class Dataset_activity(Dataset):
    def __init__(self, args=None, root_path=None, dataset='activity', device=torch.device("cpu"), q=0.016, upsampling_batch=False):
        self.data_objects = data_provider.utils.get_data(args, dataset, device, q, upsampling_batch)
        self.class_names = [0, 1, 2, 3, 4, 5, 6]