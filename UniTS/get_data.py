import math
import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import random
from physionet import PhysioNet, get_data_min_max, variable_time_collate_fn2
from sklearn import model_selection
from person_activity import PersonActivity


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

        print("batch_indices: ", batch_indices)

    return upsampled_train_data


def get_data(args, dataset, device, q=0.016, upsampling_batch=True, flag=1):
    print("upsampling_batch", upsampling_batch)
    print("args.classif", args.classif)
    print("args seed", args.seed)
    if dataset == 'P12':
        total_dataset = PhysioNet('data/physionet',
                                  quantization=q,
                                  download=True,
                                  device=device)
        PT_dict = np.load('./data/P12data/processed_data/PTdict_list.npy', allow_pickle=True)
        # arr_outcomes = np.load('./datasets/P12data/processed_data/arr_outcomes.npy', allow_pickle=True)

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
        PT_dict = np.load('./data/PAMdata/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load('./data/PAMdata/processed_data/arr_outcomes.npy', allow_pickle=True)

        total_dataset = preprocess_PAM(PT_dict, arr_outcomes)

    elif dataset == 'MIMIC':
        total_dataset = torch.load('./data/MIMIC/mimic_classification/processed/mimic.pt', map_location='cpu')
        total_dataset = [(record_id, tt, vals, mask, torch.tensor(label)) for
                         (record_id, tt, vals, mask, label) in total_dataset]

    elif dataset == 'activity':
        total_dataset = PersonActivity('datasets/activity/', n_samples = int(1e8), download=True, device = device)


    print('len(total_dataset):', len(total_dataset))
    print("total_dataset[0]:", total_dataset[0])

    global_tt = torch.unique(torch.cat([tpl[1] for tpl in total_dataset]), sorted=True)

    if dataset == 'P12':
        # get recorde_id from PTdict_list.npy
        print("idx_train[0]", idx_train[0])
        train_record_ids = [PT_dict[i]['id'] for i in idx_train]
        print("train_record_ids[0]", train_record_ids[0])
        val_record_ids = [PT_dict[i]['id'] for i in idx_val]
        test_record_ids = [PT_dict[i]['id'] for i in idx_test]

        #  dictionary mapping record_id to its tuple
        record_dict = {rec[0]: rec for rec in total_dataset}

        # get train/val/test data
        train_data = [record_dict[rid] for rid in train_record_ids]
        val_data = [record_dict[rid] for rid in val_record_ids]
        test_data = [record_dict[rid] for rid in test_record_ids]

        print("train_data[0]:", train_data[0])
        print("val_data[0]:", val_data[0])
        print("test_data[0]:", test_data[0])
    elif dataset == 'MIMIC' or dataset == 'activity':
        seen_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, random_state=args.seed,
                                                                shuffle=True)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.75, random_state=args.seed,
                                                                shuffle=False)
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
    else:
        train_data = [total_dataset[i] for i in idx_train]
        print("train_data[0]:", train_data[0])
        val_data = [total_dataset[i] for i in idx_val]
        print("val_data[0]:", val_data[0])
        test_data = [total_dataset[i] for i in idx_test]
        print("test_data[0]:", test_data[0])

    # tt: time steps, vals: observed values, mask: which values are observed
    record_id, tt, vals, mask, labels = train_data[0]

    input_dim = vals.size(-1)  # determine the number of features. vals: [T, D], where D is the number of features
    data_min, data_max = get_data_min_max(total_dataset,
                                          device)  # Compute the minimum and maximum values across all features in the entire dataset
    # batch_size = 128
    batch_size = min(len(train_data),
                     args.batch_size)  # ensures the batch size isn't larger than the dataset or user-specified number

    if flag:
        if args.classif:
            print("train len:", len(train_data))
            print("val len:", len(val_data))
            print("test len:", len(test_data))

            if upsampling_batch:
                train_data_upsamled = []
                true_labels = np.array([float(x[4].item()) for x in train_data])
                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':  # 2 classes
                    idx_0 = np.where(true_labels == 0)[0]
                    print("idx_0 length", len(idx_0))
                    idx_1 = np.where(true_labels == 1)[0]
                    print("idx_1 length", len(idx_1))
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

            if dataset == 'activity':
                test_data_combined = variable_time_collate_fn_activity(test_data, args, device, classify=args.classif, activity=True, ts_split="TEST")
                train_data_combined = variable_time_collate_fn_activity(train_data, args, device, classify=args.classif, activity=True, ts_split="TRAIN")
                val_data_combined = variable_time_collate_fn_activity(val_data, args, device, classify=args.classif, activity=True, ts_split="VAL")
            else:
                test_data_combined = variable_time_collate_fn(test_data, args, device, classify=args.classif, data_min=data_min,
                                                            data_max=data_max, global_tt=global_tt, ts_split="TEST")
                train_data_combined = variable_time_collate_fn(train_data, args, device, classify=args.classif, data_min=data_min,
                                                            data_max=data_max, global_tt=global_tt, ts_split="TRAIN")
                val_data_combined = variable_time_collate_fn(
                    val_data, args, device, classify=args.classif, data_min=data_min, data_max=data_max, global_tt=global_tt, ts_split="VAL")
            print(train_data_combined[1].sum(
            ), val_data_combined[1].sum(), test_data_combined[1].sum())
            print(train_data_combined[0].size(), train_data_combined[1].size(),
                  val_data_combined[0].size(), val_data_combined[1].size(),
                  test_data_combined[0].size(), test_data_combined[1].size())

            # convert the combined data (a tuple of data and labels) into TensorDatasets
            train_data_combined = TensorDataset(
                train_data_combined[0], train_data_combined[1].long().squeeze())
            val_data_combined = TensorDataset(
                val_data_combined[0], val_data_combined[1].long().squeeze())
            test_data_combined = TensorDataset(
                test_data_combined[0], test_data_combined[1].long().squeeze())

            print(test_data_combined[0])
        else:
            # if not classification (e.g., regression/forecasting)
            train_data_combined = variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)

        # shuffle=False since it's handled above
        train_dataloader = DataLoader(
            train_data_combined, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=batch_size, shuffle=False)

    else:
        # if flag is not set, use variable_time_collate_fn2 for custom handling
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

    data_objects = {"dataset_obj": {},
                    "train_data": train_data,
                    "train_dataloader": train_dataloader,
                    "test_data": test_data,
                    "test_dataloader": test_dataloader,
                    "input_dim": input_dim,  # number of features
                    "n_train_batches": len(train_dataloader),  # number of batches in train
                    "n_test_batches": len(test_dataloader),
                    "attr": {},  # optional
                    "classif_per_tp": False,  # (optional) boolean flag indicating classification per time point or not
                    "n_labels": 1}  # (optional) how many labels per sample are expected
    if args.classif:
        # if classification, also create and store a validation DataLoader
        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
        data_objects["val_data"] = val_data
        data_objects["val_dataloader"] = val_dataloader
    return data_objects  # return all the prepared data and metadata as a dictionary


def variable_time_collate_fn(batch, args, device=torch.device("cpu"), classify=False, activity=False,
                             data_min=None, data_max=None, global_tt=None, ts_split='TRAIN'):
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
    
    combined_tt = global_tt.to(device)

    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        indices = torch.searchsorted(combined_tt, tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        if labels is not None:
            combined_labels[b] = labels

    ts_rows = []
    ts_labels = []
    ts_lens = []
    ts_dims = D
    

    for b in range(len(batch)):
        lbl = int(combined_labels[b].item()) if classify else 0
        row = vals_mask_to_ts_row(combined_vals[b], combined_mask[b], lbl)
        ts_rows.append(row)
        ts_labels.append(lbl)
        ts_lens.append(combined_vals[b].shape[0])

    split = ts_split.upper() # TRAIN / VAL / TEST
    fname = f"{args.base_name}_{split}.ts"
    out = args.out_dir / fname
    write_ts_file(
        rows=ts_rows,
        dims=ts_dims,
        classes=list(set(ts_labels)),
        seq_lengths=ts_lens,
        out_path=out,
        problem_name=f"{args.base_name}_{split}"
    )

    combined_vals, _, _ = normalize_masked_data(combined_vals, combined_mask, att_min=data_min, att_max=data_max)

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    B = combined_vals.size(0)
    T = combined_tt.size(0)
    combined_tt = combined_tt.view(1, T, 1).expand(B, T, 1).to(device)
    combined_data = torch.cat(
        (combined_vals, combined_mask, combined_tt), 2)

    if classify:
        return combined_data, combined_labels
    else:
        return combined_data

def variable_time_collate_fn_activity(batch, args, device=torch.device("cpu"), classify=False, activity=True, data_min=None, data_max=None, ts_split='TRAIN'):
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
    N = batch[0][-1].shape[1] if activity else 1
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    if classify:
        if activity:
            combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
        else:
            combined_labels = torch.zeros([len(batch), N]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
        if classify:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:
                combined_labels[b] = labels.to(device)

    ts_rows = []
    ts_labels = []
    ts_lens = []
    ts_dims = D

    for b in range(len(batch)):
        lbl = int(combined_labels[b].item()) if classify else 0
        row = vals_mask_to_ts_row(enc_combined_vals[b], enc_combined_mask[b], lbl)
        ts_rows.append(row)
        ts_labels.append(lbl)
        ts_lens.append(enc_combined_vals[b].shape[0])

    split = ts_split.upper() # TRAIN / VAL / TEST
    fname = f"{args.base_name}_{split}.ts"
    out = args.out_dir / fname
    write_ts_file(
        rows=ts_rows,
        dims=ts_dims,
        classes=list(set(ts_labels)),
        seq_lengths=ts_lens,
        out_path=out,
        problem_name=f"{args.base_name}_{split}"
    )

    if not activity:
        enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    if classify:
        return combined_data, combined_labels
    else:
        return combined_data


def vals_mask_to_ts_row(values, msk, lbl):
    """
    Convert one (T,D) `values` tensor *and* its `mask` tensor into a single
    `.ts` row string.  Missing entries (mask==0) become '?'.
    """
    dims = []
    for d in range(values.size(1)):
        col = []
        for t in range(values.size(0)):
            if msk[t, d] == 0:
                # col.append('?')
                col.append('0')
            else:
                v = values[t, d].item()
                col.append(str(v))
        dims.append(','.join(col))
    return ':'.join(dims) + f":{lbl}"


def write_ts_file(rows, dims, classes, seq_lengths, out_path, problem_name):
    """Write *rows* (list[str]) to disk with a valid `.ts` header."""
    header = [
        f"@problemName {problem_name}",
        "@timestamps false",
        "@missing true",
        f"@univariate {'true' if dims == 1 else 'false'}",
    ]
    if dims > 1:
        header.append(f"@dimensions {dims}")

    equal_len = len(set(seq_lengths)) == 1
    header.append(f"@equalLength {'true' if equal_len else 'false'}")
    if equal_len:                         
        # if all equal, record that common L
        header.append(f"@seriesLength {seq_lengths[0]}")
    header.append("@classLabel true " + ' '.join(map(str, sorted(classes))))
    header.append("@data")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write('\n'.join(header) + '\n')
        f.write('\n'.join(rows) + '\n')

    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset', type=str, default='P12')
    parser.add_argument('--classif', action='store_true')
    parser.add_argument('--data_split_path', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--quantization', type=float, default=0.016,
                        help="Quantization on the physionet dataset.")
    parser.add_argument("--out_dir", type=str, default="./ts_export",
                        help="Where to put the .ts files")
    parser.add_argument("--base_name", type=str, default="P12")

    args = parser.parse_args()
    args.out_dir = Path(args.out_dir)
    get_data(args=args, dataset=args.dataset, device=torch.device("cpu"), q=args.quantization, upsampling_batch=False)