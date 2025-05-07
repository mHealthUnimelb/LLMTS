import os
import pickle

import utils
import numpy as np
import tarfile
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
import random
import argparse


# compute the minimum and maximum values per feature across a list of records
def get_data_min_max(records, device):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)  # vals: [T, D], n_features = D

    # for each features, extract the non-missing values from vals
    for b, (record_id, tt, vals, mask, labels) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]  # mask[:, i] == 1 selects only observed values for feature i
            # if no values are present for that feature in this sample, append inf and -inf as placeholders
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            # otherwise, compute min and max of observed values for that feature and append
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        # convert batch_min and batch_max from lists to tensors
        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        # For the first sample, initialize data_min and data_max.
        # For subsequent samples, update data_min and data_max by taking the element-wise minimum/maximum
        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    # return the computed per-feature min and max
    return data_min, data_max


class PhysioNet(object):
    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download',
    ]

    outcome_urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt',
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt',
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt',
    ]

    params = [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]

    params_dict = {k: i for i, k in enumerate(params)}

    labels = ["SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"]
    labels_dict = {k: i for i, k in enumerate(labels)}

    def __init__(self, root, download=False,
                 quantization=0.1, n_samples=None, device=torch.device("cpu")):
        # root: where data is stored
        # train: whether to load the training set or test set
        # download: whether to download if not present
        # quantization: rounding for time steps
        # n_samples: limit number of samples
        # device: device to store the data

        self.root = root
        # self.train = train
        self.device = device
        self.reduce = "average"
        self.quantization = quantization
        self.blacklist = ['140501', '150649', '140936', '143656', '141264', '145611', '142998', '147514', '142731',
                          '150309', '155655', '156254']

        if download:
            self.download()  # if download=True, attempt to download and process the dataset

        # check if processed data exists. If not, raise an error unless user sets download=True
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        # depending on the train flag, choose the appropriate processed data file
        # if self.train:
        #     data_file = self.training_file
        # else:
        #     data_file = self.test_file

        # load the data and labels from processed files
        if self.device == 'cpu':
            data_a = torch.load(os.path.join(self.processed_folder, self.set_a), map_location='cpu')
            data_b = torch.load(os.path.join(self.processed_folder, self.set_b), map_location='cpu')
            data_c = torch.load(os.path.join(self.processed_folder, self.set_c), map_location='cpu')
            # labels_a = torch.load(os.path.join(self.processed_folder, self.label_file_a), map_location='cpu')
            # labels_b = torch.load(os.path.join(self.processed_folder, self.label_file_b), map_location='cpu')
            # labels_c = torch.load(os.path.join(self.processed_folder, self.label_file_c), map_location='cpu')
        else:
            data_a = torch.load(os.path.join(self.processed_folder, self.set_a))
            data_b = torch.load(os.path.join(self.processed_folder, self.set_b))
            data_c = torch.load(os.path.join(self.processed_folder, self.set_c))
            # labels_a = torch.load(os.path.join(self.processed_folder, self.label_file_a))
            # labels_b = torch.load(os.path.join(self.processed_folder, self.label_file_b))
            # labels_c = torch.load(os.path.join(self.processed_folder, self.label_file_c))

        self.data = data_a + data_b + data_c
        # self.labels = labels_a + labels_b + labels_c
        print("data shape", len(self.data))
        # print("label shape:", len(self.labels))

        # if n_samples is specified, truncate the dataset to that number of samples
        if n_samples is not None:
            self.data = self.data[:n_samples]
            self.labels = self.labels[:n_samples]

    def download(self):
        if self._check_exists():
            return

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download outcome data
        outcomes = {}
        for url in self.outcome_urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)

            txtfile = os.path.join(self.raw_folder, filename)
            with open(txtfile) as f:
                lines = f.readlines()
                for l in lines[1:]:
                    l = l.rstrip().split(',')
                    if l[0] not in self.blacklist:
                        record_id, labels = l[0], np.array(l[1:]).astype(float)
                        outcomes[record_id] = torch.Tensor(labels).to(self.device)

        torch.save(
            outcomes,
            os.path.join(self.processed_folder, 'Outcomes.pt')
        )

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            tar.extractall(self.raw_folder)
            tar.close()

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder, filename.split('.')[0])
            patients = []
            total = 0
            for txtfile in os.listdir(dirname):
                if txtfile.split('.')[0] not in self.blacklist:
                    record_id = txtfile.split('.')[0]
                    # print("record_id", record_id)
                    with open(os.path.join(dirname, txtfile)) as f:
                        lines = f.readlines()
                        prev_time = 0
                        tt = [0.]
                        vals = [torch.zeros(len(self.params)).to(self.device)]
                        mask = [torch.zeros(len(self.params)).to(self.device)]
                        nobs = [torch.zeros(len(self.params))]
                        for l in lines[1:]:
                            total += 1
                            time, param, val = l.split(',')
                            # Time in hours
                            time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
                            # round up the time stamps (up to 6 min by default)
                            # used for speed -- we actually don't need to quantize it in Latent ODE
                            time = round(time / self.quantization) * self.quantization

                            # time in minutes
                            # time = float(time.split(':')[0]) * 60 + float(time.split(':')[1])

                            if time != prev_time:
                                tt.append(time)
                                vals.append(torch.zeros(len(self.params)).to(self.device))
                                mask.append(torch.zeros(len(self.params)).to(self.device))
                                nobs.append(torch.zeros(len(self.params)).to(self.device))
                                prev_time = time

                            if param in self.params_dict:
                                # vals[-1][self.params_dict[param]] = float(val)
                                n_observations = nobs[-1][self.params_dict[param]]
                                if self.reduce == 'average' and n_observations > 0:
                                    prev_val = vals[-1][self.params_dict[param]]
                                    new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                    vals[-1][self.params_dict[param]] = new_val
                                else:
                                    vals[-1][self.params_dict[param]] = float(val)
                                mask[-1][self.params_dict[param]] = 1
                                nobs[-1][self.params_dict[param]] += 1
                            else:
                                assert (param == 'RecordID' or param == ''), 'Read unexpected param {}'.format(param)
                    tt = torch.tensor(tt).to(self.device)
                    vals = torch.stack(vals)
                    mask = torch.stack(mask)

                    labels = None
                    if record_id in outcomes:
                        # Only training set has labels
                        labels = outcomes[record_id]
                        # Out of 5 label types provided for Physionet, take only the last one -- mortality
                        labels = labels[4]

                    patients.append((record_id, tt, vals, mask, labels))

            torch.save(
                patients,
                os.path.join(self.processed_folder,
                             filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            )

        print('Done!')

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]

            if not os.path.exists(
                    os.path.join(self.processed_folder,
                                 filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            ):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    # @property
    # def training_file(self):
    #     return 'set-a_{}.pt'.format(self.quantization)

    # @property
    # def test_file(self):
    #     return 'set-b_{}.pt'.format(self.quantization)

    @property
    def set_a(self):
        return 'set-a_{}.pt'.format(self.quantization)

    @property
    def set_b(self):
        return 'set-b_{}.pt'.format(self.quantization)

    @property
    def set_c(self):
        return 'set-c_{}.pt'.format(self.quantization)

    @property
    def label_file_a(self):
        return 'Outcomes-a.pt'

    @property
    def label_file_b(self):
        return 'Outcomes-b.pt'

    @property
    def label_file_c(self):
        return 'Outcomes-c.pt'

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Quantization: {}\n'.format(self.quantization)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str

    def visualize(self, timesteps, data, mask, plot_name):
        width = 15
        height = 15

        non_zero_attributes = (torch.sum(mask, 0) > 2).numpy()
        non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
        n_non_zero = sum(non_zero_attributes)

        mask = mask[:, non_zero_idx]
        data = data[:, non_zero_idx]

        params_non_zero = [self.params[i] for i in non_zero_idx]
        params_dict = {k: i for i, k in enumerate(params_non_zero)}

        n_col = 3
        n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
        fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecolor='white')

        # for i in range(len(self.params)):
        for i in range(n_non_zero):
            param = params_non_zero[i]
            param_id = params_dict[param]

            tp_mask = mask[:, param_id].long()

            tp_cur_param = timesteps[tp_mask == 1.]
            data_cur_param = data[tp_mask == 1., param_id]

            ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(), marker='o')
            ax_list[i // n_col, i % n_col].set_title(param)

        fig.tight_layout()
        fig.savefig(plot_name)
        plt.close(fig)


# def variable_time_collate_fn(batch, args, device=torch.device("cpu"), data_type="train",
#                              data_min=None, data_max=None):
#     """
#     Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
#         - record_id is a patient id
#         - tt is a 1-dimensional tensor containing T time values of observations.
#         - vals is a (T, D) tensor containing observed values for D variables.
#         - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
#         - labels is a list of labels for the current patient, if labels are available. Otherwise None.
#     Returns:
#         combined_tt: The union of all time observations.
#         combined_vals: (M, T, D) tensor containing the observed values.
#         combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
#     """
#     D = batch[0][2].shape[1]
#     combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
#     combined_tt = combined_tt.to(device)
#
#     offset = 0
#     combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
#     combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
#
#     combined_labels = None
#     N_labels = 1
#
#     combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
#     combined_labels = combined_labels.to(device=device)
#
#     for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
#         tt = tt.to(device)
#         vals = vals.to(device)
#         mask = mask.to(device)
#         if labels is not None:
#             labels = labels.to(device)
#
#         indices = inverse_indices[offset:offset + len(tt)]
#         offset += len(tt)
#
#         combined_vals[b, indices] = vals
#         combined_mask[b, indices] = mask
#
#         if labels is not None:
#             combined_labels[b] = labels
#
#     combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask,
#                                                       att_min=data_min, att_max=data_max)
#
#     if torch.max(combined_tt) != 0.:
#         combined_tt = combined_tt / torch.max(combined_tt)
#
#     data_dict = {
#         "data": combined_vals,
#         "time_steps": combined_tt,
#         "mask": combined_mask,
#         "labels": combined_labels}
#
#     data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
#     return data_dict


# a custom collate function that takes a batch of (record_id, tt, vals, mask, labels) and
# arranges them into a dictionary of tensors ready for model input
# data_min and data_max used for normalization
def variable_time_collate_fn2(batch, args, device=torch.device("cpu"), data_type="train",
                              data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]  # D is the number of features
    # find the length of each time series and determine the maximum length
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    # [len(batch), maxlen, D] shape accommodates the largest sequence, shorter sequences will remain zero-padded
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    # Copy each sample's tt, vals, and mask into the pre-allocated tensors, and creates a padded batch representation
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)  # current sample's time series length T
        enc_combined_tt[b, :currlen] = tt.to(
            device)  # Copy the sample's time steps into the padded tensor for the first 'currlen' positions
        enc_combined_vals[b, :currlen] = vals.to(device)  # Copy the sample's observed values into the padded tensor
        enc_combined_mask[b, :currlen] = mask.to(device)  # Copy the sample's mask into the padded tensor

    # create a unified time axis (combined_tt) by taking the union of all time points from all samples
    # inverse_indices helps map original time points to their positions in combined_tt
    # - torch.cat([ex[1] for ex in batch]) concatenates all time tensors from each sample
    # - torch.unique(..., sorted=True, return_inverse=True) returns:
    #   combined_tt: sorted unique time points across the entire batch
    #   inverse_indices: a mapping from each original time point to its index in combined_tt
    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    # Initialize combined values and masks aligned to combined_tt
    # Allocate (batch_size x combined_time_length x D) for the "combined" representation
    offset = 0  # We'll use offset to keep track of how many time points we've processed so far
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(
        device)  # [batch_size, total_unique_time_points, D]
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)  # same shape as above

    # Initialize combined_labels with NaN to indicate missing labels
    combined_labels = None
    N_labels = 1

    # creating a [batch_size, N_labels] tensor filled with NaN
    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)

    # For each sample:
    # Map its time steps (tt) into indices of combined_tt using inverse_indices.
    # Place its vals and mask into combined_vals and combined_mask at the corresponding indices.
    # If labels exist, store them in combined_labels
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # 'indices' represents where each time point in this sample maps into combined_tt
        indices = inverse_indices[offset:offset + len(tt)]
        # offset: the running index in the concatenated list of times
        # offset + len(tt): cover the time points for the current sample
        offset += len(tt)

        # Place the current sample's values and mask at the correct positions in the combined tensors
        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        if labels is not None:
            combined_labels[b] = labels

    # Normalize both the encoder combined values and the combined values with the provided min/max
    # Normalization sets values to a 0-1 range based on global min/max
    combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask,
                                                      att_min=data_min, att_max=data_max)
    enc_combined_vals, _, _ = utils.normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                          att_min=data_min, att_max=data_max)

    # Normalize time axis to a 0-1 range
    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    data_dict = {
        "enc_data": enc_combined_vals,  # [batch, maxlen, D], padded & normalized
        "enc_mask": enc_combined_mask,  # [batch, maxlen, D]
        "enc_time_steps": enc_combined_tt,  # [batch, maxlen]
        "data": combined_vals,  # [batch, #unique_times, D], fully combined & normalized
        "time_steps": combined_tt,  # [#unique_times]
        "mask": combined_mask,  # [batch, #unique_times, D]
        "labels": combined_labels}  # [batch, 1] (or [batch, N_labels])

    # Further process the batch (e.g., splitting into encoder/decoder sets, subsampling)
    data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
    return data_dict


# def variable_time_collate_fn3(batch, args, device=torch.device("cpu"), data_type="train",
#                               data_min=None, data_max=None):
#     """
#     Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
#       - record_id is a patient id
#       - tt is a 1-dimensional tensor containing T time values of observations.
#       - vals is a (T, D) tensor containing observed values for D variables.
#       - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
#       - labels is a list of labels for the current patient, if labels are available. Otherwise None.
#     Returns:
#       combined_tt: The union of all time observations.
#       combined_vals: (M, T, D) tensor containing the observed values.
#       combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
#     """
#     D = batch[0][2].shape[1]
#     len_tt = [ex[1].size(0) for ex in batch]
#     maxlen = np.max(len_tt)
#     enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
#     enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
#     enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
#     for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
#         currlen = tt.size(0)
#         enc_combined_tt[b, :currlen] = tt.to(device)
#         enc_combined_vals[b, :currlen] = vals.to(device)
#         enc_combined_mask[b, :currlen] = mask.to(device)
#
#     enc_combined_vals, _, _ = utils.normalize_masked_data(enc_combined_vals, enc_combined_mask,
#                                                           att_min=data_min, att_max=data_max)
#
#     if torch.max(enc_combined_tt) != 0.:
#         enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)
#
#     data_dict = {
#         "observed_data": enc_combined_vals,
#         "observed_tp": enc_combined_tt,
#         "observed_mask": enc_combined_mask}
#
#     return data_dict
