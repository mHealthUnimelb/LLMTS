import os

import lib.utils as utils
import numpy as np
import tarfile
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from lib.utils import get_device


# Adapted from: https://github.com/rtqichen/time-series-datasets

class PhysioNet(object):
    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download',
    ]

    outcome_urls = ['https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt',
                    'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt',
                    'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt']

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
                 quantization=None, n_samples=None, device=torch.device("cpu")):

        self.root = root
        self.reduce = "average" # Defines how to handle multiple measurements at the same timestamp. It will take the average if there are multiple observations at the same time.
        self.quantization = quantization

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        if device == torch.device("cpu"):
            data_a = torch.load(os.path.join(self.processed_folder, self.set_a), map_location='cpu')
            data_b = torch.load(os.path.join(self.processed_folder, self.set_b), map_location='cpu')
            data_c = torch.load(os.path.join(self.processed_folder, self.set_c), map_location='cpu')
            self.labels = torch.load(os.path.join(self.processed_folder, self.label_file), map_location='cpu')
        else:
            data_a = torch.load(os.path.join(self.processed_folder, self.set_a))
            data_b = torch.load(os.path.join(self.processed_folder, self.set_b))
            data_c = torch.load(os.path.join(self.processed_folder, self.set_c))
            self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))

        self.data = data_a + data_b + data_c  # a list with length 12000

        # If n_samples is set, keep only the first n_samples records
        if n_samples is not None:
            print('Total records:', len(self.data))
            self.data = self.data[:n_samples]
            self.labels = self.labels[:n_samples]

    def download(self):
        if self._check_exists():
            return

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download outcome data
        for url in self.outcome_urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)

            txtfile = os.path.join(self.raw_folder, filename)
            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                for l in lines[1:]:
                    l = l.rstrip().split(',')
                    record_id, labels = l[0], np.array(l[1:]).astype(float)
                    outcomes[record_id] = torch.Tensor(labels).to(self.device)

                torch.save(
                    labels,
                    os.path.join(self.processed_folder, filename.split('.')[0] + '.pt')
                )

        for url in self.urls:
            filename = url.rpartition('/')[2] # Extract filename (e.g., set-a.tar.gz) by splitting the URL
            download_url(url, self.raw_folder, filename, None) # Download the file to self.raw_folder
            # Open the tar archive and extract its contents into self.raw_folder
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            tar.extractall(self.raw_folder)
            tar.close()

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder, filename.split('.')[0])
            patients = []
            total = 0
            for txtfile in os.listdir(dirname):
                record_id = txtfile.split('.')[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.] # time points
                    vals = [torch.zeros(len(self.params))] # the measurement values at those times
                    mask = [torch.zeros(len(self.params))] # indicates which parameters were observed at each time
                    nobs = [torch.zeros(len(self.params))] # counts how many observations for each parameter at each time
                    for l in lines[1:]:
                        total += 1
                        time, param, val = l.split(',')
                        # Convert the time from hh:mm to a float in hours
                        time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.

                        # round up the time stamps (up to 6 min by default)
                        # used for speed -- we actually don't need to quantize it in Latent ODE
                        if (self.quantization != None and self.quantization != 0):
                            time = round(time / self.quantization) * self.quantization

                        # If this line belongs to a new time stamp, append a new entry to each of the lists
                        if time != prev_time:
                            tt.append(time)
                            # Create new zero tensors for storing values, masks and nobs at this new time
                            vals.append(torch.zeros(len(self.params)))
                            mask.append(torch.zeros(len(self.params)))
                            nobs.append(torch.zeros(len(self.params)))
                            prev_time = time

                        if param in self.params_dict:
                            # Number of observations already stored for this parameter in the current time step
                            n_observations = nobs[-1][self.params_dict[param]]
                            # have one or more observations, compute the average
                            if self.reduce == 'average' and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = float(val)
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1 # Increment the count of observations for that parameter at this time step
                        else:
                            assert (param == 'RecordID' or param == ''), 'Read unexpected param {}'.format(param)

                tt = torch.tensor(tt).to(self.device)
                vals = torch.stack(vals).to(self.device)
                mask = torch.stack(mask).to(self.device)

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
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def label_file(self):
        return 'Outcomes-a.pt'

    @property
    def set_a(self):
        return 'set-a_{}.pt'.format(self.quantization)

    @property
    def set_b(self):
        return 'set-b_{}.pt'.format(self.quantization)

    @property
    def set_c(self):
        return 'set-c_{}.pt'.format(self.quantization)

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

# compute the minimum and maximum values per feature across a list of records
def get_data_min_max(records, device):
    inf = torch.Tensor([float("Inf")])[0].to(device)

    data_min, data_max, time_max = None, None, -inf

    # tt: (T,), vals: (T, D), mask: (T, D)
    for b, (record_id, tt, vals, mask, labels) in enumerate(records):
        n_features = vals.size(-1) # vals: [T, D], n_features = D

        batch_min = []
        batch_max = []
        # for each features, extract the non-missing values from vals
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1] # mask[:, i] == 1 selects only observed values for feature i
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
        # For subsequent samples, update data_min and data_max by taking the element-wise minimum/maximum.
        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)
        # data_min hold the smallest observed value for each feature, data_max hold the largest observed value for each feature

        time_max = torch.max(time_max, tt.max()) # get the largest time across all records

    print('data_max:', data_max)
    print('data_min:', data_min)
    print('time_max:', time_max)

    # return the computed per-feature min and max
    return data_min, data_max, time_max


def get_seq_length(args, records):
    max_input_len = 0
    max_pred_len = 0
    lens = []
    for b, (record_id, tt, vals, mask) in enumerate(records):
        n_observed_tp = torch.lt(tt, args.history).sum()
        max_input_len = max(max_input_len, n_observed_tp)
        max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
        lens.append(n_observed_tp)
    lens = torch.stack(lens, dim=0)
    median_len = lens.median()

    return max_input_len, max_pred_len, median_len


def patch_variable_time_collate_fn(batch, args, device=torch.device("cpu"), data_type="train",
                                   data_min=None, data_max=None, time_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
        - record_id is a patient id
        - tt is a (T, ) tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
        - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
    Data form as input:
        batch_tt: (B, M, L_in, D) the batch contains a maximal L_in time values of observations among M patches.
        batch_vals: (B, M, L_in, D) tensor containing the observed values.
        batch_mask: (B, M, L_in, D) tensor containing 1 where values were observed and 0 otherwise.
    Data form to predict:
        flat_tt: (L_out) the batch contains a maximal L_out time values of observations.
        flat_vals: (B, L_out, D) tensor containing the observed values.
        flat_mask: (B, L_out, D) tensor containing 1 where values were observed and 0 otherwise.
    """

    D = batch[0][2].shape[1] # number of features, batch[0][2]: (T, D), batch: (record_id, tt, vals, mask, labels)

    # collect and unify all time points across the batch
    # [ex[1] for ex in batch] extracts tt (time arrays) from each sample in the batch
    # combined_tt is a 1D tensor of all unique time points across the entire batch
    # inverse_indices map each sample's time steps back to the positions in combined_tt
    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)

    # the number of observed time points
    # split time points into "observed" vs. "predicted"
    # args.history indicates the cutoff time. For example, we might only consider times < history as “input” or “observed.”
    # torch.lt(combined_tt, args.history) creates a boolean mask: True where combined_tt < history, False otherwise
    # .sum() counts how many time points satisfy that condition
    # observed_tp = combined_tt[:n_observed_tp]: slice out only the earliest time points that are considered “observed.”
    n_observed_tp = torch.lt(combined_tt, args.history).sum()
    observed_tp = combined_tt[:n_observed_tp]  # (n_observed_tp, )

    # divide the "observed" time range into patches
    patch_indices = []
    st, ed = 0, args.patch_size # st: start time, ed: end time
    # iterates over the total number of patches (npatch)
    for i in range(args.npatch):
        # Finds the indices in observed_tp that fall between st and ed
        if (i == args.npatch - 1):
            inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
        else:
            inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
        # collects these indices (the subset of time steps belonging to this patch).
        patch_indices.append(inds)
        st += args.stride
        ed += args.stride
    # By the end, patch_indices is a list of length npatch, each containing the indices of observed_tp belonging to that patch

    offset = 0 # This will help us track how to map each sample’s time indices (inverse_indices) back into combined_tt
    # combined_vals: (B, T_all, D), T_all = len(combined_tt), the total number of unique time points across the entire batch
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

    combined_labels = None
    N_labels = 1
    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)

    # predicted_tp, predicted_data, predicted_mask: store the future portion (time >= args.history) for each sample,
    # which we’ll later pad into a batch
    predicted_tp = []
    predicted_data = []
    predicted_mask = []
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        # For sample b, we slice out only the portion corresponding to its own time series of length len(tt).
        # This tells us where each tt[i] fits in combined_tt
        indices = inverse_indices[offset:offset + len(tt)]
        # increments offset so the next sample will pick up the next chunk of indices in inverse_indices
        offset += len(tt)
        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        # split each sample's datainto "observed" vs.  "predicted"
        tmp_n_observed_tp = torch.lt(tt, args.history).sum() # The number of time points in tt that are earlier than args.history
        predicted_tp.append(tt[tmp_n_observed_tp:]) # The remaining time points (tt[tmp_n_observed_tp:]) are the “future” or “predicted” portion
        predicted_data.append(vals[tmp_n_observed_tp:])
        predicted_mask.append(mask[tmp_n_observed_tp:])

    # truncate the combined arrays to only the "observed" portion
    # combined_tt originally contained all unique time points (observed + future),
    # we only want to keep the first n_observed_tp time points that are < args.history
    combined_tt = combined_tt[:n_observed_tp]
    combined_vals = combined_vals[:, :n_observed_tp]
    combined_mask = combined_mask[:, :n_observed_tp]

    # pad_sequence(list_of_tensors, batch_first=True) will produce a tensor of shape (B, T_max, ...),
    # where T_max is the max length among all sample segments. The shorter ones get padded with zeros
    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)

    if (args.dataset != 'ushcn'):
        # normalize data
        combined_vals = utils.normalize_masked_data(combined_vals, combined_mask,
                                                    att_min=data_min, att_max=data_max)
        predicted_data = utils.normalize_masked_data(predicted_data, predicted_mask,
                                                     att_min=data_min, att_max=data_max)

    # normalize the time dimension combined_tt and predicted_tp from [0, time_max] to [0, 1],
    # so the model can handle time more easily
    combined_tt = utils.normalize_masked_tp(combined_tt, att_min=0, att_max=time_max)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min=0, att_max=time_max)

    data_dict = {
        "data": combined_vals,  # (n_batch, T_o, D), the observed values
        "time_steps": combined_tt,  # (T_o, ), the truncated and normalized time points
        "mask": combined_mask,  # (n_batch, T_o, D), the observed-data mask
        "data_to_predict": predicted_data, # The future portion
        "tp_to_predict": predicted_tp, # The future portion
        "mask_predicted_data": predicted_mask, # The future portion
    }


    data_dict = utils.split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)

    return data_dict


def variable_time_collate_fn(batch, args, device=torch.device("cpu"), data_type="train",
                             data_min=None, data_max=None, time_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
        - record_id is a patient id
        - tt is a (T, ) tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
    Returns:
        batch_tt: (B, L) the batch contains a maximal L time values of observations.
        batch_vals: (B, L, D) tensor containing the observed values.
        batch_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
    """

    observed_tp = []
    observed_data = []
    observed_mask = []
    predicted_tp = []
    predicted_data = []
    predicted_mask = []

    for b, (record_id, tt, vals, mask) in enumerate(batch):
        n_observed_tp = torch.lt(tt, args.history).sum()
        observed_tp.append(tt[:n_observed_tp])
        observed_data.append(vals[:n_observed_tp])
        observed_mask.append(mask[:n_observed_tp])

        predicted_tp.append(tt[n_observed_tp:])
        predicted_data.append(vals[n_observed_tp:])
        predicted_mask.append(mask[n_observed_tp:])

    observed_tp = pad_sequence(observed_tp, batch_first=True)
    observed_data = pad_sequence(observed_data, batch_first=True)
    observed_mask = pad_sequence(observed_mask, batch_first=True)
    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)

    if (args.dataset != 'ushcn'):
        observed_data = utils.normalize_masked_data(observed_data, observed_mask,
                                                    att_min=data_min, att_max=data_max)
        predicted_data = utils.normalize_masked_data(predicted_data, predicted_mask,
                                                     att_min=data_min, att_max=data_max)

    observed_tp = utils.normalize_masked_tp(observed_tp, att_min=0, att_max=time_max)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min=0, att_max=time_max)

    data_dict = {"observed_data": observed_data,
                 "observed_tp": observed_tp,
                 "observed_mask": observed_mask,
                 "data_to_predict": predicted_data,
                 "tp_to_predict": predicted_tp,
                 "mask_predicted_data": predicted_mask,
                 }

    return data_dict


if __name__ == '__main__':
    torch.manual_seed(1991)

    dataset = PhysioNet('../data/physionet', train=False, download=True)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn)
    print(dataloader.__iter__().next())
