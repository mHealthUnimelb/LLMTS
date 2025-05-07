# pylint: disable=E1101
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, confusion_matrix

import numpy as np
from physionet import PhysioNet, get_data_min_max, variable_time_collate_fn2
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score
from person_activity import PersonActivity
import random


def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


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


def evaluate(dim, rec, dec, test_loader, args, num_sample=10, device="cuda"):
    mse, test_n = 0.0, 0.0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)
            observed_data, observed_mask, observed_tp = (
                test_batch[:, :, :dim],
                test_batch[:, :, dim: 2 * dim],
                test_batch[:, :, -1],
            )
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean, qz0_logvar = (
                out[:, :, : args.latent_dim],
                out[:, :, args.latent_dim:],
            )
            epsilon = torch.randn(
                num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            batch, seqlen = observed_tp.size()
            time_steps = (
                observed_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
            )
            pred_x = dec(z0, time_steps)
            pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
            pred_x = pred_x.mean(0)
            mse += mean_squared_error(observed_data, pred_x, observed_mask) * batch
            test_n += batch
    return mse / test_n


def compute_losses(dim, dec_train_batch, qz0_mean, qz0_logvar, pred_x, args, device):
    observed_data, observed_mask \
        = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2 * dim]

    noise_std = args.std
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    if args.norm:
        logpx /= observed_mask.sum(-1).sum(-1)
        analytic_kl /= observed_mask.sum(-1).sum(-1)
    return logpx, analytic_kl


def evaluate_classifier(model, test_loader, dec=None, args=None, classifier=None,
                        dim=41, device='cuda', reconst=False, num_sample=1, dataset='P12'):
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2 * dim], test_batch[:, :, -1]
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                       :args.latent_dim], out[:, :, args.latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if args.classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    out = classifier(pred_x)
                else:
                    out = classifier(z0)
            if args.classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
                preds = out.softmax(dim=1) # probilities
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len * num_sample
                preds = out.softmax(dim=1)
        pred.append(preds.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    true = np.concatenate(true, 0)

    acc = np.mean(pred.argmax(1) == true)

    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'MIMIC':
        auc = metrics.roc_auc_score(true, pred[:, 1]) if not args.classify_pertp else 0.
        aupr = average_precision_score(true, pred[:, 1]) if not args.classify_pertp else 0.
        precision = precision_score(true, pred.argmax(1)) if not args.classify_pertp else 0.
        recall = recall_score(true, pred.argmax(1)) if not args.classify_pertp else 0.
        F1 = 2 * (precision * recall) / (precision + recall) if not args.classify_pertp else 0.
        return test_loss / pred.shape[0], acc, auc, aupr, precision, recall, F1
    elif dataset == 'activity':
        auc = metrics.roc_auc_score(one_hot(true), pred)
        aupr = average_precision_score(one_hot(true), pred)
        precision = precision_score(true, pred.argmax(1), average='macro', )
        recall = recall_score(true, pred.argmax(1), average='macro', )
        F1 = 2 * (precision * recall) / (precision + recall)
        return test_loss / pred.shape[0], acc, auc, aupr, precision, recall, F1
    elif dataset == 'PAM':
        auc = metrics.roc_auc_score(one_hot(true), pred) if not args.classify_pertp else 0.
        aupr = average_precision_score(one_hot(true), pred) if not args.classify_pertp else 0.
        precision = precision_score(true, pred.argmax(1), average='macro', ) if not args.classify_pertp else 0.
        recall = recall_score(true, pred.argmax(1), average='macro', ) if not args.classify_pertp else 0.
        F1 = 2 * (precision * recall) / (precision + recall) if not args.classify_pertp else 0.
        return test_loss / pred.shape[0], acc, auc, aupr, precision, recall, F1


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

    '''
    # calculate and save P19 statistics - age, gender, density scores (can be used for all algorithms)
    idx_under_65 = []
    idx_over_65 = []
    idx_male = []
    idx_female = []

    for i in range(len(PT_dict)):
        if total[i][0] == PT_dict[i]['id']:
            age, gender, _, _, _, _ = PT_dict[i]['extended_static']
            if age > 0:
                if age < 65:
                    idx_under_65.append(i)
                else:
                    idx_over_65.append(i)
            if gender == 0:
                idx_female.append(i)
            if gender == 1:
                idx_male.append(i)

    np.save('P19_idx_under_65.npy', np.array(idx_under_65), allow_pickle=True)
    np.save('P19_idx_over_65.npy', np.array(idx_over_65), allow_pickle=True)
    np.save('P19_idx_male.npy', np.array(idx_male), allow_pickle=True)
    np.save('P19_idx_female.npy', np.array(idx_female), allow_pickle=True)

    # save density scores
    X_features = np.array([d['arr'] for d in PT_dict])
    counts = np.count_nonzero(X_features, axis=(0, 1))
    ascending_indices = np.argsort(counts)
    density_scores = counts / (X_features.shape[0] * 60)
    res = [[ind, density_scores[ind], labels_ts[:-1][ind]] for ind in ascending_indices]
    np.save('P19_density_scores.npy', res, allow_pickle=True)
    '''

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

    '''
    # calculate and save P19 statistics - gender, density scores (can be used for all algorithms)
    idx_male = []
    idx_female = []
    for i in range(len(PT_dict)):
        if total[i][0] == str(i):
            vec = PT_dict[i]['extended_static']
            if vec[-3] > 0:
                idx_female.append(i)
            if vec[-4] > 0:
                idx_male.append(i)

    print('\nOnly 1.329/36.443 samples have gender data available.\n')

    np.save('eICU_idx_male.npy', np.array(idx_male), allow_pickle=True)
    np.save('eICU_idx_female.npy', np.array(idx_female), allow_pickle=True)

    # save density scores
    X_features = np.array([d['arr'] for d in PT_dict])
    counts = np.count_nonzero(X_features, axis=(0, 1))
    ascending_indices = np.argsort(counts)
    density_scores = counts / (X_features.shape[0] * 300)
    res = [[ind, density_scores[ind], labels_ts[ind]] for ind in ascending_indices]
    np.save('eICU_density_scores.npy', res, allow_pickle=True)
    '''

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


def get_data(args, dataset, device, q, upsampling_batch, split_type, feature_removal_level, missing_ratio, flag=1,
             reverse=False, predictive_label='mortality'):
    if dataset == 'P12':
        train_dataset_obj_1 = PhysioNet('data/physionet', train=True,
                                        quantization=q,
                                        download=True, n_samples=12000,
                                        device=device, set_letter='a')

        train_dataset_obj_2 = PhysioNet('data/physionet', train=True,
                                        quantization=q,
                                        download=True, n_samples=12000,
                                        device=device, set_letter='b')

        train_dataset_obj_3 = PhysioNet('data/physionet', train=True,
                                        quantization=q,
                                        download=True, n_samples=12000,
                                        device=device, set_letter='c')

        PT_dict = np.load('./data/P12data/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load('./data/P12data/processed_data/arr_outcomes.npy', allow_pickle=True)
        # total_dataset = preprocess_P12(PT_dict, arr_outcomes, q)
        idx_train, idx_val, idx_test = np.load(args.data_split_path, allow_pickle=True)

        dataset_1 = train_dataset_obj_1[:len(train_dataset_obj_1)]
        dataset_2 = train_dataset_obj_2[:len(train_dataset_obj_2)]
        dataset_3 = train_dataset_obj_3[:len(train_dataset_obj_3)]

        total_dataset = dataset_1 + dataset_2 + dataset_3
        print("total_dataset len", len(total_dataset))

        if predictive_label == 'LoS':
            los_outcomes = np.load('../saved/LoS_y1_out.npy', allow_pickle=True)
            for i, tpl in enumerate(total_dataset):
                a, b, c, d, _ = tpl
                los_label = los_outcomes[i][0]
                los_label = torch.tensor(los_label, dtype=torch.float32)
                total_dataset[i] = (a, b, c, d, los_label)

        '''
        # calculate and save statistics
        idx_under_65 = []
        idx_over_65 = []
        idx_male = []
        idx_female = []

        P_list = np.load('P_list.npy', allow_pickle=True)

        for i in range(len(P_list)):
            if total_dataset[i][0] == P_list[i]['id']:
                age, gender, _, _, _ = P_list[i]['static']
                if age > 0:
                    if age < 65:
                        idx_under_65.append(i)
                    else:
                        idx_over_65.append(i)
                if gender == 0:
                    idx_female.append(i)
                if gender == 1:
                    idx_male.append(i)

        np.save('mtand_idx_under_65.npy', np.array(idx_under_65), allow_pickle=True)
        np.save('mtand_idx_over_65.npy', np.array(idx_over_65), allow_pickle=True)
        np.save('mtand_idx_male.npy', np.array(idx_male), allow_pickle=True)
        np.save('mtand_idx_female.npy', np.array(idx_female), allow_pickle=True)
        '''

    elif dataset == 'P19':
        PT_dict = np.load('../../../P19data/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        labels_ts = np.load('../../../P19data/processed_data/labels_ts.npy', allow_pickle=True)
        labels_demogr = np.load('../../../P19data/processed_data/labels_demogr.npy', allow_pickle=True)
        arr_outcomes = np.load('../../../P19data/processed_data/arr_outcomes_6.npy', allow_pickle=True)

        total_dataset = preprocess_P19(PT_dict, arr_outcomes, labels_ts)

    elif dataset == 'eICU':
        PT_dict = np.load('../../../eICUdata/processed_data/PTdict_list.npy', allow_pickle=True)
        labels_ts = np.load('../../../eICUdata/processed_data/eICU_ts_vars.npy', allow_pickle=True)
        labels_demogr = np.load('../../../eICUdata/processed_data/eICU_static_vars.npy', allow_pickle=True)
        arr_outcomes = np.load('../../../eICUdata/processed_data/arr_outcomes.npy', allow_pickle=True)

        total_dataset = preprocess_eICU(PT_dict, arr_outcomes, labels_ts)

    elif dataset == 'PAM':
        PT_dict = np.load('../../../PAMdata/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load('../../../PAMdata/processed_data/arr_outcomes.npy', allow_pickle=True)

        total_dataset = preprocess_PAM(PT_dict, arr_outcomes)

    elif dataset == 'MIMIC':
        total_dataset = torch.load('./data/MIMIC/mimic_classification/processed/mimic.pt', map_location='cpu')
        total_dataset = [(record_id, tt, vals, mask, torch.tensor(label, dtype=torch.long)) for
                         (record_id, tt, vals, mask, label) in total_dataset]
        
    elif dataset == 'activity':
        # args.pred_window = 1000
        total_dataset = PersonActivity('data/activity/', n_samples = int(1e8), download=True, device = device)
        # total_dataset = torch.load('./data/activiaty/processed/data.pt', map_location='cpu')

    print('len(total_dataset):', len(total_dataset))

    # train_data = []
    # val_data = []
    # test_data = []
    #
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

        print("train_data[0]:", train_data[0])
        print("val_data[0]:", val_data[0])
        print("test_data[0]:", test_data[0])
    elif dataset == 'MIMIC' or dataset == 'activity':
        print("seed", args.seed)
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

    # if split_type == 'random':
    #     # Shuffle and split
    #     train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.9,     # 80% train, 10% validation, 10% test
    #                                                              shuffle=True)

    record_id, tt, vals, mask, labels = train_data[0]

    input_dim = vals.size(-1)
    data_min, data_max = get_data_min_max(total_dataset, device)
    batch_size = 128

    if flag:
        if args.classif:
            print("train len:", len(train_data))
            print("val len:", len(val_data))
            print("test len:", len(test_data))

            # if split_type == 'random':
            #     train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8889, shuffle=True)  # 80% train, 10% validation, 10% test
            # elif split_type == 'age' or split_type == 'gender':
            #     val_data, test_data = model_selection.train_test_split(test_data, train_size=0.5, shuffle=False)
            #
            # if dataset == 'P12':
            #     num_all_features = 36
            # elif dataset == 'P19':
            #     num_all_features = 34
            # elif dataset == 'eICU':
            #     num_all_features = 14
            # elif dataset == 'PAM':
            #     num_all_features = 17
            #
            # num_missing_features = round(missing_ratio * num_all_features)
            # if feature_removal_level == 'sample':
            #     for i, tpl in enumerate(val_data):
            #         idx = np.random.choice(num_all_features, num_missing_features, replace=False)
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         val_data[i] = tuple(tpl)
            #     for i, tpl in enumerate(test_data):
            #         idx = np.random.choice(num_all_features, num_missing_features, replace=False)
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         test_data[i] = tuple(tpl)
            # elif feature_removal_level == 'set':
            #     if dataset == 'P12':
            #         dict_params = train_dataset_obj_1.params_dict
            #         density_scores_names = np.load('../saved/IG_density_scores_P12.npy', allow_pickle=True)[:, 1]
            #         idx = [dict_params[name] for name in density_scores_names[:num_missing_features]]
            #     elif dataset == 'P19':
            #         labels_ts = np.load('../../../P19data/processed_data/labels_ts.npy', allow_pickle=True)
            #         dict_params = {label: i for i, label in enumerate(labels_ts[:-1])}
            #         density_scores_names = np.load('../saved/IG_density_scores_P19.npy', allow_pickle=True)[:, 1]
            #         idx = [dict_params[name] for name in density_scores_names[:num_missing_features]]
            #     elif dataset == 'eICU':
            #         labels_ts = np.load('../../../eICUdata/processed_data/eICU_ts_vars.npy', allow_pickle=True)
            #         dict_params = {label: i for i, label in enumerate(labels_ts)}
            #         density_scores_names = np.load('../saved/IG_density_scores_eICU.npy', allow_pickle=True)[:, 1]
            #         idx = [dict_params[name] for name in density_scores_names[:num_missing_features]]
            #     elif dataset == 'PAM':
            #         density_scores_indices = np.load('../saved/IG_density_scores_PAM.npy', allow_pickle=True)[:, 0]
            #         idx = list(map(int, density_scores_indices[:num_missing_features]))
            #
            #     for i, tpl in enumerate(val_data):
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         val_data[i] = tuple(tpl)
            #     for i, tpl in enumerate(test_data):
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         test_data[i] = tuple(tpl)

            if upsampling_batch:
                train_data_upsamled = []
                true_labels = np.array([float(x[4].item()) for x in train_data])
                # true_labels = np.array(list(map(lambda x: float(x[7]), np.array(train_data)[:, 4])))
                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'MIMIC':  # 2 classes
                    idx_0 = np.where(true_labels == 0)[0]
                    idx_1 = np.where(true_labels == 1)[0]
                    for _ in range(len(true_labels) // batch_size):
                        indices = random_sample(idx_0, idx_1, batch_size)
                        for i in indices:
                            train_data_upsamled.append(train_data[i])
                    # train_data_upsamled = balanced_batch_sampler(train_data, true_labels, batch_size, 2)
                elif dataset == 'PAM':  # 8 classes
                    # for b in range(len(true_labels) // batch_size):
                    #     indices = random_sample_8(true_labels, batch_size)
                    #     for i in indices:
                    #         train_data_upsamled.append(train_data[i])
                    train_data_upsamled = balanced_batch_sampler(train_data, true_labels, batch_size, 8)

                train_data = train_data_upsamled

            if dataset == 'activity':
                test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif, activity=True)
                train_data_combined = variable_time_collate_fn(train_data, device, classify=args.classif, activity=True)
                val_data_combined = variable_time_collate_fn(val_data, device, classify=args.classif, activity=True)
            else:
                test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif, data_min=data_min,
                                                            data_max=data_max)
                train_data_combined = variable_time_collate_fn(train_data, device, classify=args.classif, data_min=data_min,
                                                            data_max=data_max)
                val_data_combined = variable_time_collate_fn(
                    val_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            print(train_data_combined[1].sum(
            ), val_data_combined[1].sum(), test_data_combined[1].sum())
            print(train_data_combined[0].size(), train_data_combined[1].size(),
                  val_data_combined[0].size(), val_data_combined[1].size(),
                  test_data_combined[0].size(), test_data_combined[1].size())

            train_data_combined = TensorDataset(
                train_data_combined[0], train_data_combined[1].long().squeeze())
            val_data_combined = TensorDataset(
                val_data_combined[0], val_data_combined[1].long().squeeze())
            test_data_combined = TensorDataset(
                test_data_combined[0], test_data_combined[1].long().squeeze())
        else:
            train_data_combined = variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)

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

    data_objects = {"dataset_obj": {},
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader),
                    "attr": {},  # optional
                    "classif_per_tp": False,  # optional
                    "n_labels": 1}  # optional
    if args.classif:
        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
        data_objects["val_dataloader"] = val_dataloader
    return data_objects


def variable_time_collate_fn(batch, device=torch.device("cpu"), classify=False, activity=False,
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
                if labels is not None:
                    combined_labels[b] = labels.to(device)

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


def get_activity_data(args, device):
    n_samples = min(10000, args.n)
    dataset_obj = PersonActivity('data/PersonActivity',
                                 download=True, n_samples=n_samples, device=device)

    print(dataset_obj)

    train_data, test_data = model_selection.train_test_split(dataset_obj, train_size=0.8,
                                                             random_state=42, shuffle=True)

    record_id, tt, vals, mask, labels = train_data[0]
    input_dim = vals.size(-1)

    batch_size = min(min(len(dataset_obj), args.batch_size), args.n)
    test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif,
                                                  activity=True)
    train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                            random_state=11, shuffle=True)
    train_data_combined = variable_time_collate_fn(
        train_data, device, classify=args.classif, activity=True)
    val_data_combined = variable_time_collate_fn(
        val_data, device, classify=args.classif, activity=True)
    print(train_data_combined[1].sum(
    ), val_data_combined[1].sum(), test_data_combined[1].sum())
    print(train_data_combined[0].size(), train_data_combined[1].size(),
          val_data_combined[0].size(), val_data_combined[1].size(),
          test_data_combined[0].size(), test_data_combined[1].size())

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long())
    val_data_combined = TensorDataset(
        val_data_combined[0], val_data_combined[1].long())
    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long())

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=batch_size, shuffle=False)

    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "val_dataloader": val_dataloader,
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader),
                    "classif_per_tp": False,  # optional
                    "n_labels": 1}  # optional

    return data_objects


def irregularly_sampled_data_gen(n=10, length=20, seed=0):
    np.random.seed(seed)
    obs_values, ground_truth, obs_times = [], [], []
    for i in range(n):
        t1 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        t2 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        t3 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        a = 10 * np.random.randn()
        b = 10 * np.random.rand()
        f1 = .8 * np.sin(20 * (t1 + a) + np.sin(20 * (t1 + a))) + \
             0.01 * np.random.randn()
        f2 = -.5 * np.sin(20 * (t2 + a + 20) + np.sin(20 * (t2 + a + 20))
                          ) + 0.01 * np.random.randn()
        f3 = np.sin(12 * (t3 + b)) + 0.01 * np.random.randn()
        obs_times.append(np.stack((t1, t2, t3), axis=0))
        obs_values.append(np.stack((f1, f2, f3), axis=0))
        t = np.linspace(0, 1, 100)
        fg1 = .8 * np.sin(20 * (t + a) + np.sin(20 * (t + a)))
        fg2 = -.5 * np.sin(20 * (t + a + 20) + np.sin(20 * (t + a + 20)))
        fg3 = np.sin(12 * (t + b))
        ground_truth.append(np.stack((fg1, fg2, fg3), axis=0))
    return obs_values, ground_truth, obs_times


def sine_wave_data_gen(args, seed=0):
    np.random.seed(seed)
    obs_values, ground_truth, obs_times = [], [], []
    for _ in range(args.n):
        t = np.sort(np.random.choice(np.linspace(
            0, 1., 101), size=args.length, replace=True))
        b = 10 * np.random.rand()
        f = np.sin(12 * (t + b)) + 0.1 * np.random.randn()
        obs_times.append(t)
        obs_values.append(f)
        tc = np.linspace(0, 1, 100)
        fg = np.sin(12 * (tc + b))
        ground_truth.append(fg)

    obs_values = np.array(obs_values)
    obs_times = np.array(obs_times)
    ground_truth = np.array(ground_truth)
    print(obs_values.shape, obs_times.shape, ground_truth.shape)
    mask = np.ones_like(obs_values)
    combined_data = np.concatenate((np.expand_dims(obs_values, axis=2), np.expand_dims(
        mask, axis=2), np.expand_dims(obs_times, axis=2)), axis=2)
    print(combined_data.shape)
    print(combined_data[0])
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": 1,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def kernel_smoother_data_gen(args, alpha=100., seed=0, ref_points=10):
    np.random.seed(seed)
    obs_values, ground_truth, obs_times = [], [], []
    for _ in range(args.n):
        key_values = np.random.randn(ref_points)
        key_points = np.linspace(0, 1, ref_points)

        query_points = np.sort(np.random.choice(
            np.linspace(0, 1., 101), size=args.length, replace=True))
        weights = np.exp(-alpha * (np.expand_dims(query_points,
                                                  1) - np.expand_dims(key_points, 0)) ** 2)
        weights /= weights.sum(1, keepdims=True)
        query_values = np.dot(weights, key_values)
        obs_values.append(query_values)
        obs_times.append(query_points)

        query_points = np.linspace(0, 1, 100)
        weights = np.exp(-alpha * (np.expand_dims(query_points,
                                                  1) - np.expand_dims(key_points, 0)) ** 2)
        weights /= weights.sum(1, keepdims=True)
        query_values = np.dot(weights, key_values)
        ground_truth.append(query_values)

    obs_values = np.array(obs_values)
    obs_times = np.array(obs_times)
    ground_truth = np.array(ground_truth)
    print(obs_values.shape, obs_times.shape, ground_truth.shape)
    mask = np.ones_like(obs_values)
    combined_data = np.concatenate((np.expand_dims(obs_values, axis=2), np.expand_dims(
        mask, axis=2), np.expand_dims(obs_times, axis=2)), axis=2)
    print(combined_data.shape)
    print(combined_data[0])
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": 1,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def get_toy_data(args):
    dim = 3
    obs_values, ground_truth, obs_times = irregularly_sampled_data_gen(
        args.n, args.length)
    obs_times = np.array(obs_times).reshape(args.n, -1)
    obs_values = np.array(obs_values)
    combined_obs_values = np.zeros((args.n, dim, obs_times.shape[-1]))
    mask = np.zeros((args.n, dim, obs_times.shape[-1]))
    for i in range(dim):
        combined_obs_values[:, i, i *
                                  args.length: (i + 1) * args.length] = obs_values[:, i]
        mask[:, i, i * args.length: (i + 1) * args.length] = 1.
    combined_data = np.concatenate(
        (combined_obs_values, mask, np.expand_dims(obs_times, axis=1)), axis=1)
    combined_data = np.transpose(combined_data, (0, 2, 1))
    print(combined_data.shape)
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": dim,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def compute_pertp_loss(label_predictions, true_label, mask):
    criterion = nn.CrossEntropyLoss(reduction='none')
    n_traj, n_tp, n_dims = label_predictions.size()
    label_predictions = label_predictions.reshape(n_traj * n_tp, n_dims)
    true_label = true_label.reshape(n_traj * n_tp, n_dims)
    mask = torch.sum(mask, -1) > 0
    mask = mask.reshape(n_traj * n_tp, 1)
    _, true_label = true_label.max(-1)
    ce_loss = criterion(label_predictions, true_label.long())
    ce_loss = ce_loss * mask
    return torch.sum(ce_loss) / mask.sum()


def get_physionet_data_extrap(args, device, q, flag=1):
    train_dataset_obj = PhysioNet('data/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=min(10000, args.n),
                                  device=device)
    # Use custom collate_fn to combine samples with arbitrary time observations.
    # Returns the dataset along with mask and time steps
    test_dataset_obj = PhysioNet('data/physionet', train=False,
                                 quantization=q,
                                 download=True, n_samples=min(10000, args.n),
                                 device=device)

    # Combine and shuffle samples from physionet Train and physionet Test
    total_dataset = train_dataset_obj[:len(train_dataset_obj)]

    if not args.classif:
        # Concatenate samples from original Train and Test sets
        # Only 'training' physionet samples are have labels.
        # Therefore, if we do classifiction task, we don't need physionet 'test' samples.
        total_dataset = total_dataset + \
                        test_dataset_obj[:len(test_dataset_obj)]
    print(len(total_dataset))
    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                             random_state=42, shuffle=True)

    record_id, tt, vals, mask, labels = train_data[0]

    input_dim = vals.size(-1)
    data_min, data_max = get_data_min_max(total_dataset, device)
    batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)

    def extrap(test_data):
        enc_test_data = []
        dec_test_data = []
        for (record_id, tt, vals, mask, labels) in test_data:
            midpt = 0
            for tp in tt:
                if tp < 24:
                    midpt += 1
                else:
                    break
            if mask[:midpt].sum() and mask[midpt:].sum():
                enc_test_data.append(
                    (record_id, tt[:midpt], vals[:midpt], mask[:midpt], labels))
                dec_test_data.append(
                    (record_id, tt[midpt:], vals[midpt:], mask[midpt:], labels))
        return enc_test_data, dec_test_data

    enc_train_data, dec_train_data = extrap(train_data)
    enc_test_data, dec_test_data = extrap(test_data)
    enc_train_data_combined = variable_time_collate_fn(
        enc_train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    dec_train_data_combined = variable_time_collate_fn(
        dec_train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    enc_test_data_combined = variable_time_collate_fn(
        enc_test_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    dec_test_data_combined = variable_time_collate_fn(
        dec_test_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    print(enc_train_data_combined.shape, dec_train_data_combined.shape)
    print(enc_test_data_combined.shape, dec_test_data_combined.shape)

    # keep the timepoints in enc between 0.0 and 0.5
    enc_train_data_combined[:, :, -1] *= 0.5
    enc_test_data_combined[:, :, -1] *= 0.5
    print(enc_train_data_combined[0, :, -1], dec_train_data_combined[0, :, -1])
    enc_train_dataloader = DataLoader(
        enc_train_data_combined, batch_size=batch_size, shuffle=False)
    dec_train_dataloader = DataLoader(
        dec_train_data_combined, batch_size=batch_size, shuffle=False)
    enc_test_dataloader = DataLoader(
        enc_test_data_combined, batch_size=batch_size, shuffle=False)
    dec_test_dataloader = DataLoader(
        dec_test_data_combined, batch_size=batch_size, shuffle=False)

    attr_names = train_dataset_obj.params
    data_objects = {"dataset_obj": train_dataset_obj,
                    "enc_train_dataloader": enc_train_dataloader,
                    "enc_test_dataloader": enc_test_dataloader,
                    "dec_train_dataloader": dec_train_dataloader,
                    "dec_test_dataloader": dec_test_dataloader,
                    "input_dim": input_dim,
                    "attr": attr_names,  # optional
                    "classif_per_tp": False,  # optional
                    "n_labels": 1}  # optional

    return data_objects


def subsample_timepoints(data, time_steps, mask, percentage_tp_to_sample=None):
    # Subsample percentage of points from each time series
    for i in range(data.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * percentage_tp_to_sample)
        subsampled_idx = sorted(np.random.choice(
            non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

        data[i, tp_to_set_to_zero] = 0.
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask
