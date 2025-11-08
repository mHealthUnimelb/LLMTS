from math import inf
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from sklearn import model_selection
import pandas as pd
from tqdm import tqdm
from data_utils.person_activity import PersonActivity
from data_utils.physionet import PhysioNet
import pickle
import random

Constants_PAD = 0


def get_data_min_max(records, device):
    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals).to(device))
                batch_max.append(torch.max(non_missing_vals).to(device))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max


def proc_hii_data(x, y, input_dim, args):
    x = x[:, :input_dim * 2 + 1]

    if args.debug_flag:
        x = x[:1000, :]
        y = y[:1000]

    if args.task == "los":
        y = y - 1

    x = np.transpose(x, (0, 2, 1))

    new_x = np.empty((len(x), len(x[1]), input_dim * 3 + 1))

    print("data preprocessing in batch...")
    total = len(x)
    batch_sz = 20000

    pbar = tqdm(range(0, total, batch_sz))
    for start in pbar:
        end = min(start + batch_sz, total)

        new_x[start:end, :, :input_dim * 2 + 1] = process_data(x[start:end], input_dim)
        new_x[start:end, :, input_dim * 2 + 1:input_dim * 3 + 1] = cal_tau(x[start:end, :, -1],
                                                                           x[start:end, :, input_dim:2 * input_dim])

    print("data preprocess in batch done.")
    return new_x, y


def proc_hii_set_data(x, y, input_dim, args):
    x = x[:, :input_dim * 2 + 1]

    if args.debug_flag:
        x = x[:1000, :]
        y = y[:1000]

    if args.task == "los":
        y = y - 1

    x = np.transpose(x, (0, 2, 1))

    if args.enc_tau is not None:
        new_x = np.empty((len(x), len(x[1]), input_dim * 3 + 1))
    else:
        new_x = np.empty((len(x), len(x[1]), input_dim * 2 + 1))

    print("data preprocessing in batch...")
    total = len(x)
    batch_sz = 100

    pbar = tqdm(range(0, total, batch_sz))
    for start in pbar:
        end = min(start + batch_sz, total)

        new_x[start:end, :, :input_dim * 2 + 1] = process_data(x[start:end], input_dim)

        if args.enc_tau is not None:
            new_x[start:end, :, input_dim * 2 + 1:input_dim * 3 + 1] = cal_tau(x[start:end, :, -1],
                                                                               x[start:end, :, input_dim:2 * input_dim])

    print("data preprocess in batch done.")
    return new_x, y


def get_clints_hii_data(args, to_set=False):
    if args.task == "vent" or args.task == "vaso":
        data_folder_x = args.root_path + args.data_path + 'cip/'
        data_folder_y = args.root_path + args.data_path + 'cip/' + args.task + '_'
    elif args.task == "pretrain":
        data_folder_x = args.root_path + args.data_path + 'mor/'
        data_folder_y = args.root_path + args.data_path + 'mor/'
    else:
        data_folder_x = args.root_path + args.data_path + args.task + '/'
        data_folder_y = args.root_path + args.data_path + args.task + '/'

    dataloader = []

    for set_name in ['train', 'val', 'test']:
        data_x_all = []
        data_y_all = []

        if set_name == 'train':
            shuffle = True
        else:
            shuffle = False

        print("loading " + set_name + " data")
        if set_name == "train" and args.task != "mor" and args.load_in_batch:
            for i in range(5):
                data_x = np.load(data_folder_x + set_name + '_input' + str(i) + '.npy', allow_pickle=True)
                data_y = np.load(data_folder_y + set_name + '_output' + str(i) + '.npy', allow_pickle=True)

                args.num_types = int((data_x.shape[1] - 1) / 2)
                data_x, data_y = proc_hii_data(data_x, data_y, args.num_types, args)
                data_x_all.append(data_x)
                data_y_all.append(data_y)
                del data_x, data_y

            data_x_all = np.concatenate(data_x_all)
            data_y_all = np.concatenate(data_y_all)

        else:
            data_y_all = np.load(data_folder_y + set_name + '_output.npy', allow_pickle=True)
            data_x_all = np.load(data_folder_x + set_name + '_input.npy', allow_pickle=True)

            args.num_types = int((data_x_all.shape[1] - 1) / 2)
            data_x_all, data_y_all = proc_hii_data(data_x_all, data_y_all, args.num_types, args)

        print(data_x_all.shape, data_y_all.shape)
        dataloader.append(get_data_loader(data_x_all, data_y_all, args, shuffle=shuffle))
        del data_x_all, data_y_all

    print("type num: ", args.num_types)
    return dataloader[0], dataloader[1], dataloader[2]


def get_data_loader(data_x, data_y, args, shuffle=False):
    data_combined = TensorDataset(torch.from_numpy(data_x).float(),
                                  torch.from_numpy(data_y).long().squeeze())
    dataloader = DataLoader(
        data_combined, batch_size=args.batch_size, shuffle=shuffle, num_workers=8)

    return dataloader


def cal_tau(observed_tp, observed_mask):
    # input [B,L,K], [B,L]
    # return [B,L,K]
    # observed_mask, observed_tp = x[:, :, input_dim:2 * input_dim], x[:, :, -1]
    if observed_tp.ndim == 2:
        tmp_time = observed_mask * np.expand_dims(observed_tp, axis=-1)  # [B,L,K]
    else:
        tmp_time = observed_tp.copy()

    b, l, k = tmp_time.shape

    new_mask = observed_mask.copy()
    new_mask[:, 0, :] = 1
    tmp_time[new_mask == 0] = np.nan
    tmp_time = tmp_time.transpose((1, 0, 2))  # [L,B,K]
    tmp_time = np.reshape(tmp_time, (l, b * k))  # [L, B*K]

    # padding the missing value with the next value
    df1 = pd.DataFrame(tmp_time)
    df1 = df1.fillna(method='ffill')
    tmp_time = np.array(df1)

    tmp_time = np.reshape(tmp_time, (l, b, k))
    tmp_time = tmp_time.transpose((1, 0, 2))  # [B,L,K]

    tmp_time[:, 1:] -= tmp_time[:, :-1]
    del new_mask
    return tmp_time * observed_mask


def process_data(x, input_dim, m=None, tt=None, x_only=False):
    if not x_only:
        observed_vals, observed_mask, observed_tp = x[:, :,
                                                    :input_dim], x[:, :, input_dim:2 * input_dim], x[:, :, -1]
        observed_tp = np.expand_dims(observed_tp, axis=-1)
    else:
        observed_vals = x
        assert m is not None
        observed_mask = m
        observed_tp = tt

    observed_vals = tensorize_normalize(observed_vals)
    observed_vals[observed_mask == 0] = 0
    if not x_only:
        return np.concatenate((observed_vals, observed_mask, observed_tp), axis=-1)
    return observed_vals


def tensorize_normalize(P_tensor):
    mf, stdf = getStats(P_tensor)
    P_tensor = normalize(P_tensor, mf, stdf)
    return P_tensor


def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        if len(vals_f) > 0:
            mf[f] = np.mean(vals_f)
            tmp_std = np.std(vals_f)
            stdf[f] = np.max([tmp_std, eps])
    return mf, stdf


def normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    return Pnorm_tensor


def variable_time_collate_fn(batch, device, input_dim, return_np=False, to_set=False, maxlen=None,
                             data_min=None, data_max=None, activity=False):
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
    D = batch[0][2].shape[1]
    # number of labels
    # N = batch[0][-1].shape[1] if activity else 1
    if maxlen is None:
        len_tt = [ex[1].size(0) for ex in batch]
        maxlen = np.max(len_tt)

    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)

    if activity:
        combined_labels = torch.zeros([len(batch), maxlen]).to(device)
    else:
        combined_labels = torch.zeros([len(batch)]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = min(tt.size(0), maxlen)
        enc_combined_tt[b, :currlen] = tt[:currlen].to(device)
        enc_combined_vals[b, :currlen] = vals[:currlen].to(device)
        enc_combined_mask[b, :currlen] = mask[:currlen].to(device)

        if labels.dim() == 2:
            combined_labels[b] = torch.argmax(labels, dim=-1)
        else:
            combined_labels[b] = labels.to(device)

    enc_combined_vals = torch.tensor(process_data(
        enc_combined_vals.cpu().numpy(),
        m=enc_combined_mask.cpu().numpy(),
        tt=enc_combined_tt,
        input_dim=input_dim,
        x_only=True)).to(enc_combined_tt.device)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    tau = torch.tensor(cal_tau(enc_combined_tt.cpu().numpy(), enc_combined_mask.cpu().numpy())).to(
        enc_combined_vals.device)
    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1), tau), 2)

    return combined_data, combined_labels


def get_activity_data(args, device):
    n_samples = 8000
    dataset_obj = PersonActivity(args.data_path + 'PersonActivity',
                                 download=True, n_samples=n_samples, device=device)

    print(dataset_obj)

    train_data, test_data = model_selection.train_test_split(dataset_obj, train_size=0.8,
                                                             random_state=args.seed, shuffle=False)

    record_id, tt, vals, mask, labels = train_data[0]
    input_dim = vals.size(-1)
    args.num_types = input_dim

    batch_size = min(len(dataset_obj), args.batch_size)

    if not args.retrain:
        train_data, val_data = model_selection.train_test_split(train_data, train_size=0.75,
                                                                random_state=args.seed, shuffle=False)

        val_data_combined = variable_time_collate_fn(val_data, device, input_dim=input_dim, activity=True)
        val_data_combined = TensorDataset(
            val_data_combined[0], val_data_combined[1].long())

        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
    else:
        val_dataloader = None

    train_data_combined = variable_time_collate_fn(train_data, device, input_dim=input_dim, activity=True)
    test_data_combined = variable_time_collate_fn(test_data, device, input_dim=input_dim, activity=True)

    # norm_mean = train_data_combined[0][:, :, :input_dim].mean(dim=0, keepdim=True).cpu()

    print(train_data_combined[1].sum(), test_data_combined[1].sum())

    print(train_data_combined[0].size(), train_data_combined[1].size(),
          test_data_combined[0].size(), test_data_combined[1].size())

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long())
    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long())

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, input_dim


def get_physionet_data(args, device, q=0.016, flag=1):
    train_dataset_obj = PhysioNet(args.data_path + '/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=8000,
                                  device='cpu')
    #   device=device)

    # Combine and shuffle samples from physionet Train and physionet Test
    total_dataset = train_dataset_obj[:len(train_dataset_obj)]
    data_min, data_max = get_data_min_max(total_dataset, device)
    print(len(total_dataset))

    # For reproduce experimental results
    with open('./data_utils/train_record_id.pkl', 'rb') as f:
        train_record_id = pickle.load(f)

    with open('./data_utils/test_record_id.pkl', 'rb') as f:
        test_record_id = pickle.load(f)

    train_data = [item for item in total_dataset if item[0] in train_record_id]
    train_data = sorted(train_data, key=lambda x: train_record_id.index(x[0]))

    test_data = [item for item in total_dataset if item[0] in test_record_id]

    # Shuffle and split
    # train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, 
    #                                                          random_state=42, shuffle=True)

    _, _, vals, _, _ = train_data[0]

    input_dim = vals.size(-1)
    batch_size = min(len(train_dataset_obj), args.batch_size)
    args.num_types = input_dim

    if not args.retrain:
        train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                                random_state=11, shuffle=False)

        val_data_combined = variable_time_collate_fn(val_data, device, input_dim=input_dim, data_min=data_min,
                                                     data_max=data_max)

        val_data_combined = TensorDataset(
            val_data_combined[0], val_data_combined[1].long().squeeze())

        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
    else:
        val_dataloader = None

    train_data_combined = variable_time_collate_fn(train_data, device, input_dim=input_dim, data_min=data_min,
                                                   data_max=data_max)
    test_data_combined = variable_time_collate_fn(test_data, device, input_dim=input_dim, data_min=data_min,
                                                  data_max=data_max)

    print(train_data_combined[1].sum(
    ), test_data_combined[1].sum())
    print(train_data_combined[0].size(), train_data_combined[1].size(),
          test_data_combined[0].size(), test_data_combined[1].size())

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long().squeeze())

    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long().squeeze())

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, input_dim


def preprocess_P12(PT_dict, arr_outcomes, quantization):
    """
    Process a list of patient records (PT_dict) and outcome values (arr_outcomes).
    Each patient record is assumed to have:
      - 'id': a record identifier (string)
      - 'static': a tuple of 5 static variables
      - 'arr': a numpy.ndarray of shape (T, 36) for T time steps
      - 'time': a numpy.ndarray of shape (T, 1) with time stamps (dynamic times)
      - 'length': the number T of valid time steps in arr and time

    The output for each patient is a tuple:
      (record_id, tt, vals, mask, outcome)
    where:
      - tt is a 1D tensor of time stamps. A new initial time stamp 0 is prepended for the static row.
      - vals is a 2D tensor of shape ((length + 1), 41) where the first row is the static data
        (5 static variables and 36 zeros) and the remaining rows are from 'arr' padded on the left
        with 5 zeros (static variables).
      - mask is built in a similar fashion: for the static row, the mask is 1 for the first 5 entries
        and 0 for the remaining 36; for dynamic rows, we compute the nonzero indicator from 'arr' and
        pad with 5 zeros on the left (static variables).
      - outcome is a tensor converted from arr_outcomes.
    """
    total = []
    for i, patient in enumerate(PT_dict):
        length = patient['length']
        record_id = patient['id']

        # # process static features (time = 0)
        # static_features = torch.tensor(patient['static'], dtype=torch.float32)  # shape: (5,)
        # static_row = torch.cat([static_features, torch.zeros(36, dtype=torch.float32)])  # shape: (41,)
        #
        # # For dynamic features, get the measurement array and pad with 5 zeros at the beginning.
        # arr_dynamic = torch.tensor(patient['arr'][:length, :], dtype=torch.float32)  # shape: (length, 36)
        # dynamic_vals = torch.cat([torch.zeros((length, 5), dtype=torch.float32), arr_dynamic],
        #                          dim=1)  # shape: (length, 41)
        #
        # # concatenate static and dynamic features
        # vals = torch.cat([static_row.unsqueeze(0), dynamic_vals], dim=0)  # shape: (length+1, 41)

        # prepare the values array of shape [length+1, 5 (static) + 36 (dynamic) = 41]
        vals = torch.zeros((length + 1, 41), dtype=torch.float32)

        # fill row 0 (time = 0) with static variables in columns 0..4
        static_vars = torch.tensor(patient['static'], dtype=torch.float32)  # shape [5]
        vals[0, :5] = static_vars

        # fill rows [1..length] in columns [5..40] with the dynamic features
        dynamic_vars = torch.tensor(patient['arr'][:length, :], dtype=torch.float32)  # shape [length, 36]
        vals[1:, 5:] = dynamic_vars

        # tt = torch.squeeze(torch.tensor(patient['time'][:length]), 1)
        dynamic_tt = torch.squeeze(torch.tensor(patient['time'][:length]), 1)
        tt = torch.zeros(length + 1, dtype=torch.float32)
        tt[1:] = dynamic_tt
        # vals = torch.tensor(patient['arr'][:length, :], dtype=torch.float32)

        # convert time in minutes
        tt = tt / 60.0
        # round each time according to the quantization
        tt = torch.round(tt / quantization) * quantization

        # # dynamic features
        # m = np.zeros(shape=patient['arr'][:length, :].shape)
        # m[patient['arr'][:length, :].nonzero()] = 1
        # dynamic_mask = torch.tensor(m, dtype=torch.float32)  # shape: (length, 36)
        # dynamic_mask = torch.cat([torch.zeros((length, 5), dtype=torch.float32), dynamic_mask],
        #                          dim=1)  # shape: (length, 41)
        # # static mask
        # static_mask = torch.cat([torch.ones(5, dtype=torch.float32), torch.zeros(36, dtype=torch.float32)])
        # mask = torch.cat([static_mask.unsqueeze(0), dynamic_mask], dim=0)  # shape: (length+1, 41)

        # mask
        mask = torch.zeros((length + 1, 41), dtype=torch.float32)

        # row 0, columns 0..4 are the static variables (mark these as present)
        mask[0, :5] = 1.0

        # for the time-series portion, copy the nonzero positions
        arr_np = patient['arr'][:length, :]  # shape [length, 36]
        mask_np = np.zeros_like(arr_np)
        mask_np[arr_np.nonzero()] = 1
        # put mask_np into columns [5..40] of rows [1..length]
        mask[1:, 5:] = torch.tensor(mask_np, dtype=torch.float32)

        # mask = torch.tensor(m, dtype=torch.float32)
        outcome = torch.tensor(arr_outcomes[i][-1], dtype=torch.float32)
        total.append((record_id, tt, vals, mask, outcome))

    return total


def preprocess_P19(PT_dict, arr_outcomes, labels_ts):
    total = []
    for i, patient in enumerate(PT_dict):
        length = patient['length']
        record_id = patient['id']
        tt = torch.squeeze(torch.tensor(patient['time'][:length]), 1)
        vals = torch.tensor(patient['arr'][:length, :], dtype=torch.float32)
        m = np.zeros(shape=patient['arr'][:length, :].shape)
        m[patient['arr'][:length, :].nonzero()] = 1
        mask = torch.tensor(m, dtype=torch.float32)
        outcome = torch.tensor(arr_outcomes[i][0], dtype=torch.float32)
        total.append((record_id, tt, vals, mask, outcome))

    return total


def preprocess_eICU(PT_dict, arr_outcomes, labels_ts):
    total = []
    for i, patient in enumerate(PT_dict):
        record_id = str(i)
        tt = torch.squeeze(torch.tensor(patient['time']), 1)
        vals = torch.tensor(patient['arr'], dtype=torch.float32)
        m = np.zeros(shape=patient['arr'].shape)
        m[patient['arr'].nonzero()] = 1
        mask = torch.tensor(m, dtype=torch.float32)
        outcome = torch.tensor(arr_outcomes[i], dtype=torch.float32)
        total.append((record_id, tt, vals, mask, outcome))

    return total


def preprocess_PAM(PT_dict, arr_outcomes):
    length = 600
    total = []
    for i, patient in enumerate(PT_dict):
        record_id = str(i)
        tt = torch.tensor(list(range(length)))
        vals = torch.tensor(patient, dtype=torch.float32)
        m = np.zeros(shape=patient.shape)
        m[patient.nonzero()] = 1
        mask = torch.tensor(m, dtype=torch.float32)
        outcome = torch.tensor(arr_outcomes[i][0], dtype=torch.float32)
        total.append((record_id, tt, vals, mask, outcome))
    return total


def random_sample(idx_0, idx_1, batch_size):
    """
    Returns a balanced sample by randomly sampling without replacement.

    :param idx_0: indices of negative samples
    :param idx_1: indices of positive samples
    :param batch_size: batch size
    :return: indices of balanced batch of negative and positive samples
    """
    idx0_batch = np.random.choice(idx_0, size=int(batch_size / 2), replace=False)
    idx1_batch = np.random.choice(idx_1, size=int(batch_size / 2), replace=False)
    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
    return idx


def random_sample_8(ytrain, B, replace=False):
    """ Returns a balanced sample of tensors by randomly sampling without replacement. """
    idx0_batch = np.random.choice(np.where(ytrain == 0)[0], size=int(B / 8), replace=replace)
    idx1_batch = np.random.choice(np.where(ytrain == 1)[0], size=int(B / 8), replace=replace)
    idx2_batch = np.random.choice(np.where(ytrain == 2)[0], size=int(B / 8), replace=replace)
    idx3_batch = np.random.choice(np.where(ytrain == 3)[0], size=int(B / 8), replace=replace)
    idx4_batch = np.random.choice(np.where(ytrain == 4)[0], size=int(B / 8), replace=replace)
    idx5_batch = np.random.choice(np.where(ytrain == 5)[0], size=int(B / 8), replace=replace)
    idx6_batch = np.random.choice(np.where(ytrain == 6)[0], size=int(B / 8), replace=replace)
    idx7_batch = np.random.choice(np.where(ytrain == 7)[0], size=int(B / 8), replace=replace)
    idx = np.concatenate(
        [idx0_batch, idx1_batch, idx2_batch, idx3_batch, idx4_batch, idx5_batch, idx6_batch, idx7_batch], axis=0)
    return idx


def balanced_batch_sampler(train_data, true_labels, batch_size, n_classes):
    """
        Creates an upsampled training dataset with balanced batches.

        Each batch contains an equal number of samples from each class.
        Samples are drawn randomly without immediate repetition. When the
        available pool for a class is exhausted, it is refilled and reshuffled.

        Args:
            train_data (list or array): List of training samples.
            true_labels (np.array): Array of labels corresponding to train_data.
            batch_size (int): Total batch size; must be divisible by n_classes.
            n_classes (int): Number of classes.

        Returns:
            list: Upsampled training data with balanced batches.
        """
    # Ensure batch_size is divisible by the number of classes
    if batch_size % n_classes != 0:
        raise ValueError("batch_size must be divisible by n_classes")

    # Number of samples per class per batch
    per_class_per_batch = batch_size // n_classes

    # Create a dictionary for the full list of indices for each class
    class_indices = {}
    # Also maintain an available pool for each class from which samples are drawn
    available_indices = {}
    for cls in range(n_classes):
        indices = np.where(true_labels == cls)[0].tolist()
        class_indices[cls] = indices
        available_indices[cls] = indices.copy()
        # np.random.shuffle(available_indices[cls])

    # Decide on the number of iterations of batches to generate in this epoch.
    # Here, we ensure the total upsampled data covers at least the size of the original dataset.
    num_iter_batch = int(np.ceil(len(true_labels) / batch_size))

    upsampled_train_data = []

    for _ in range(num_iter_batch):
        batch_indices = []
        for cls in range(n_classes):
            sampled = []
            # Use leftover samples first if available.
            num_available = len(available_indices[cls])
            if num_available >= per_class_per_batch:
                # Enough available samples: take the first 'per_class_per_batch' samples.
                sampled = available_indices[cls][:per_class_per_batch]
                available_indices[cls] = available_indices[cls][per_class_per_batch:]
            else:
                # Not enough samples remaining: use all the available samples.
                if num_available > 0:
                    sampled = available_indices[cls].copy()
                    available_indices[cls] = []
                # Calculate how many additional samples are needed.
                needed = per_class_per_batch - len(sampled)
                # Refill the pool by shuffling a complete copy of the class indices.
                new_pool = class_indices[cls].copy()
                np.random.shuffle(new_pool)
                additional_samples = new_pool[:needed]
                sampled.extend(additional_samples)
                # Store the remaining samples in the new pool for future use.
                available_indices[cls] = new_pool[needed:]
            batch_indices.extend(sampled)

        # Optionally shuffle the combined batch indices for randomness within the batch
        # np.random.shuffle(batch_indices)
        # Append the samples corresponding to these indices to the upsampled training data
        for idx in batch_indices:
            upsampled_train_data.append(train_data[idx])

    return upsampled_train_data


def get_data(args, dataset, device, q=0.016, upsampling_batch=False):
    if dataset == 'P12':
        total_dataset = PhysioNet('../datasets/physionet',
                                  quantization=q,
                                  download=True,
                                  device=device)
        PT_dict = np.load('../datasets/P12data/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load('../datasets/P12data/processed_data/arr_outcomes.npy', allow_pickle=True)
        idx_train, idx_val, idx_test = np.load(args.data_split_path, allow_pickle=True)
    elif dataset == 'P19':
        PT_dict = np.load('../P19data/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        labels_ts = np.load('../P19data/processed_data/labels_ts.npy', allow_pickle=True)
        labels_demogr = np.load('../P19data/processed_data/labels_demogr.npy', allow_pickle=True)
        arr_outcomes = np.load('../P19data/processed_data/arr_outcomes_6.npy', allow_pickle=True)

        total_dataset = preprocess_P19(PT_dict, arr_outcomes, labels_ts)
    elif dataset == 'eICU':
        PT_dict = np.load('../../../eICUdata/processed_data/PTdict_list.npy', allow_pickle=True)
        labels_ts = np.load('../../../eICUdata/processed_data/eICU_ts_vars.npy', allow_pickle=True)
        labels_demogr = np.load('../../../eICUdata/processed_data/eICU_static_vars.npy', allow_pickle=True)
        arr_outcomes = np.load('../../../eICUdata/processed_data/arr_outcomes.npy', allow_pickle=True)

        total_dataset = preprocess_eICU(PT_dict, arr_outcomes, labels_ts)

    elif dataset == 'PAM':
        PT_dict = np.load('../datasets/PAMdata/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load('../datasets/PAMdata/processed_data/arr_outcomes.npy', allow_pickle=True)

        total_dataset = preprocess_PAM(PT_dict, arr_outcomes)

    elif dataset == 'MIMIC':
        total_dataset = torch.load('../datasets/MIMIC/mimic_classification/processed/mimic.pt', map_location='cpu')
        total_dataset = [(record_id, tt, vals, mask, torch.tensor(label)) for
                         (record_id, tt, vals, mask, label) in total_dataset]

    if dataset == 'P12':
        # get recorde_id from PTdict_list.npy
        train_record_ids = [PT_dict[i]['id'] for i in idx_train]
        val_record_ids = [PT_dict[i]['id'] for i in idx_val]
        test_record_ids = [PT_dict[i]['id'] for i in idx_test]

        #  dictionary mapping record_id to its tuple
        record_dict = {rec[0]: rec for rec in total_dataset}

        # get train/val/test data
        train_data = [record_dict[rid] for rid in train_record_ids]
        val_data = [record_dict[rid] for rid in val_record_ids]
        test_data = [record_dict[rid] for rid in test_record_ids]
    elif dataset == 'MIMIC':
        seen_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, random_state=args.seed,
                                                                shuffle=True)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.75, random_state=args.seed,
                                                                shuffle=False)
    else:
        train_data = [total_dataset[i] for i in idx_train]
        val_data = [total_dataset[i] for i in idx_val]
        test_data = [total_dataset[i] for i in idx_test]

    # few-shot learning
    if args.percent is not None:
        subset_len = int(len(train_data) * (args.percent/100))
        train_data = train_data[:subset_len]

    # tt: time steps, vals: observed values, mask: which values are observed
    record_id, tt, vals, mask, labels = train_data[0]

    input_dim = vals.size(-1)  # determine the number of features. vals: [T, D], where D is the number of features
    data_min, data_max = get_data_min_max(total_dataset,
                                          device)  # Compute the minimum and maximum values across all features in the entire dataset
    batch_size = min(len(train_data),
                     args.batch_size)  # ensures the batch size isn't larger than the dataset or user-specified number
    args.num_types = input_dim

    if not args.retrain:
        if upsampling_batch:
            train_data_upsamled = []
            true_labels = np.array([float(x[4].item()) for x in train_data])
            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'MIMIC':  # 2 classes
                idx_0 = np.where(true_labels == 0)[0]
                idx_1 = np.where(true_labels == 1)[0]
                # Method 1
                # for _ in range(len(true_labels) // batch_size):
                #     indices = random_sample(idx_0, idx_1, batch_size)
                #     for i in indices:
                #         train_data_upsamled.append(train_data[i])

                # Method 2
                train_data_upsamled = balanced_batch_sampler(train_data, true_labels, batch_size, 2)

            elif dataset == 'PAM':  # 8 classes
                # for b in range(len(true_labels) // batch_size):
                #     indices = random_sample_8(true_labels, batch_size)
                #     for i in indices:
                #         train_data_upsamled.append(train_data[i])
                train_data_upsamled = balanced_batch_sampler(train_data, true_labels, batch_size, 8)

            train_data = train_data_upsamled

        val_data_combined = variable_time_collate_fn(val_data, device, input_dim=input_dim, data_min=data_min,
                                                     data_max=data_max)

        val_data_combined = TensorDataset(
            val_data_combined[0], val_data_combined[1].long().squeeze())

        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
    else:
        val_dataloader = None

    train_data_combined = variable_time_collate_fn(train_data, device, input_dim=input_dim, data_min=data_min,
                                                   data_max=data_max)
    test_data_combined = variable_time_collate_fn(test_data, device, input_dim=input_dim, data_min=data_min,
                                                  data_max=data_max)

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long().squeeze())

    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long().squeeze())

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, input_dim



def cal_label_freq(labels):
    freq_dict = {}
    for i in labels:
        if i not in freq_dict:
            freq_dict[i] = sum(labels == i)

    print(freq_dict)


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max
