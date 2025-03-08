import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

import lib.utils as utils
from torch.distributions import uniform

from torch.utils.data import DataLoader
from lib.physionet import *
from lib.ushcn import *
from lib.mimic import MIMIC
from lib.person_activity import *
from sklearn import model_selection


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices):
    # This function takes a batch of time series data (already partially collated) and reorganizes it into smaller
    # patches of time. The main goal is to reshape the observed portion (B, T_obs, D) into (B, Npatch, PatchLen, D)
    # using a carefully constructed index mapping, while also respecting the varying number of non-missing
    # observations within each patch
    # data_dict: A dictionary typically produced by a collate function such as patch_variable_time_collate_fn. It must have keys like:
    #           "time_steps": Observed times (shape: (T_obs,) or (B, T_obs)).
    #           "data": Observed data (shape: (B, T_obs, D)).
    #           "mask": Observed data mask (shape: (B, T_obs, D)).
    #           "tp_to_predict", "data_to_predict", "mask_predicted_data": Future/prediction portion
    # args: Contains hyperparameters (e.g., npatch, patch indexing info, etc.)
    # n_observed_tp: Number of observed time points (<= args.history)
    # patch_indices: A list (of length args.npatch) where each element is a torch.Tensor of indices describing
    # which time points belong to that patch (e.g., [start_idx, ..., end_idx] within [0..n_observed_tp-1])

    device = get_device(data_dict["data"])

    split_dict = {"tp_to_predict": data_dict["tp_to_predict"].clone(),
                  "data_to_predict": data_dict["data_to_predict"].clone(),
                  "mask_predicted_data": data_dict["mask_predicted_data"].clone()
                  }

    observed_tp = data_dict["time_steps"].clone()  # (n_observed_tp, )
    observed_data = data_dict["data"].clone()  # (bs, n_observed_tp, D)
    observed_mask = data_dict["mask"].clone()  # (bs, n_observed_tp, D)

    # n_batch = B (the batch size).
    # n_tp = T_obs (the number of observed time points).
    # n_dim = D (the number of features)
    n_batch, n_tp, n_dim = observed_data.shape
    # Tile out the data to shape (B, Npatch, T_obs, D)
    observed_tp_patches = observed_tp.view(1, 1, -1, 1).repeat(n_batch, args.npatch, 1, n_dim)
    observed_data_patches = observed_data.view(n_batch, 1, n_tp, n_dim).repeat(1, args.npatch, 1, 1)
    observed_mask_patches = observed_mask.view(n_batch, 1, n_tp, n_dim).repeat(1, args.npatch, 1, 1)

    max_patch_len = 0 # track how many non-missing data points are in each patch
    for i in range(args.npatch):
        indices = patch_indices[i] # indices is a 1D tensor of time indices belonging to patch i
        if (len(indices) == 0): continue
        st_ind, ed_ind = indices[0], indices[-1] # start/end time indices for that patch
        # observed_mask[:, st_ind:ed_ind+1] is the sub-mask for all batch samples,
        # focusing on only that patch’s times. Shape is (B, patch_width, D)
        # .sum(dim=1) sums over the time dimension, giving (B, D).
        # Then .max() finds the maximum across the batch dimension, returning a single scalar.
        # This scalar is how many non-missing points (across time) the “fullest” sample has within that patch
        # .item() converts that scalar tensor to a Python float/int
        n_data_points = observed_mask[:, st_ind:ed_ind + 1].sum(dim=1).max().item()
        # keeps track of the largest number of non-missing points in any patch for any sample
        max_patch_len = max(max_patch_len, int(n_data_points))

    observed_mask_patches_fill = torch.zeros_like(observed_mask_patches,
                                                  dtype=observed_mask.dtype)  # n_batch, npacth, n_tp, n_dim
    # A (B, Npatch, max_patch_len, D) tensor initialized with n_tp (which is effectively an “out of range” index).
    # This will become the final “index map” that tells us which time-step index each “slot” should pull from.
    patch_indices_fianl = torch.full((n_batch, args.npatch, max_patch_len, n_dim), n_tp).to(
        device)  # n_batch, npacth, max_patch_len, n_dim
    # Another (B, npatch, max_patch_len, D) zero tensor to keep track of which “slots” are valid or not
    observed_mask_patches_fill_reindex = torch.zeros_like(patch_indices_fianl, dtype=observed_mask.dtype)
    # Shape (B, max_patch_len, D), where each row is [0, 1, 2, ...] up to max_patch_len-1. This is used to compare
    # against how many non-missing points are in each patch, effectively deciding how many slots to fill
    aux_tensor = torch.arange(max_patch_len).view(1, max_patch_len, 1).repeat(n_batch, 1, n_dim).to(device)
    for i in range(args.npatch):
        indices = patch_indices[i]
        if (len(indices) == 0): continue
        st_ind, ed_ind = indices[0], indices[-1]
        # Copy the original mask portion (B, patch_width, D) into the big zero array at
        # [batch, patch_i, time_range, dim]. This indicates which times are truly part of patch i
        observed_mask_patches_fill[:, i, st_ind:ed_ind + 1] = observed_mask[:, st_ind:ed_ind + 1, :]
        # For each sample in the batch and each feature, sum how many non-missing entries are in [st_ind..ed_ind].
        # The shape is (B, 1, D).
        L = observed_mask[:, st_ind:ed_ind + 1, :].sum(dim=1, keepdim=True)  # (bs, 1, D)
        # For each (B, D), if aux_tensor (ranging from 0..max_patch_len-1) is < L, that slot is True,
        # meaning it’s a valid “observed data” slot. If aux_tensor is >= L, that slot is False.
        # This effectively says: if you have L=5 non-missing points, only the first 5 slots
        # along aux_tensor should be filled as valid data
        observed_mask_patches_fill_reindex[:, i] = (aux_tensor < L)  # let first L[i] to be True

    ### return a indices tuple like ([...], [...], [...], [...])
    # mask_inds = torch.nonzero(..., as_tuple=True): Finds all positions where observed_mask_patches_fill_reindex.permute(0,1,3,2) != 0
    # permute(0,1,3,2) changes the dimension order from (B, Npatch, max_patch_len, D) to (B, Npatch, D, max_patch_len)
    # torch.nonzero(..., as_tuple=True) yields a tuple of index tensors (one for each dimension). Let's call them (batch_ids, patch_ids, dim_ids, slot_ids)
    mask_inds = torch.nonzero(observed_mask_patches_fill_reindex.permute(0, 1, 3, 2), as_tuple=True)  # reset indices

    ind_values = torch.nonzero(observed_mask_patches_fill.permute(0, 1, 3, 2), as_tuple=True)[
        -1]  # original indices of dimension 2

    ### fill n_tp if the number of observed points are less than max_patch_len
    patch_indices_fianl.index_put_((mask_inds[0], mask_inds[1], mask_inds[3], mask_inds[2]), ind_values)

    pad_zeros_data = torch.zeros([n_batch, args.npatch, 1, n_dim]).to(device)
    observed_tp_patches = torch.cat([observed_tp_patches, pad_zeros_data], dim=2).gather(2,
                                                                                         patch_indices_fianl)  # (n_batch, npatch, max_patch_len, n_dim)
    observed_data_patches = torch.cat([observed_data_patches, pad_zeros_data], dim=2).gather(2, patch_indices_fianl)
    observed_mask_patches = torch.cat([observed_mask_patches, pad_zeros_data], dim=2).gather(2, patch_indices_fianl)

    split_dict["observed_tp"] = observed_tp_patches
    split_dict["observed_data"] = observed_data_patches
    split_dict["observed_mask"] = observed_mask_patches

    return split_dict


#####################################################################################################
def parse_datasets(args, patch_ts=False, length_stat=False):
    device = args.device
    dataset_name = args.dataset

    ##################################################################
    ### PhysioNet dataset ###
    ### MIMIC dataset ###
    if dataset_name in ["physionet", "mimic"]:

        ### list of tuples (record_id, tt, vals, mask) ###
        if dataset_name == "physionet":
            total_dataset = PhysioNet('../data/physionet', quantization=args.quantization,
                                      download=True, n_samples=args.n, device=device)
        elif dataset_name == "mimic":
            total_dataset = MIMIC('../data/mimic/', n_samples=args.n, device=device)

        # Shuffle and split
        seen_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, random_state=42,
                                                                shuffle=True)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.75, random_state=42,
                                                                shuffle=False)
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
        test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
        print("Test record ids (first 20):", test_record_ids[:20])
        print("Test record ids (last 20):", test_record_ids[-20:])

        record_id, tt, vals, mask, labels = train_data[0]

        input_dim = vals.size(-1)

        batch_size = min(min(len(seen_data), args.batch_size), args.n)
        data_min, data_max, time_max = get_data_min_max(seen_data, device)  # (n_dim,), (n_dim,)

        if (patch_ts):
            collate_fn = patch_variable_time_collate_fn
        else:
            collate_fn = variable_time_collate_fn

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda batch: collate_fn(batch, args, device, data_type="train",
                                                                          data_min=data_min, data_max=data_max,
                                                                          time_max=time_max))
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                    collate_fn=lambda batch: collate_fn(batch, args, device, data_type="val",
                                                                        data_min=data_min, data_max=data_max,
                                                                        time_max=time_max))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: collate_fn(batch, args, device, data_type="test",
                                                                         data_min=data_min, data_max=data_max,
                                                                         time_max=time_max))

        data_objects = {
            "train_dataloader": utils.inf_generator(train_dataloader),
            "val_dataloader": utils.inf_generator(val_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "input_dim": input_dim,
            "n_train_batches": len(train_dataloader),
            "n_val_batches": len(val_dataloader),
            "n_test_batches": len(test_dataloader),
            # "attr": total_dataset.params, #optional
            "data_max": data_max,  # optional
            "data_min": data_min,
            "time_max": time_max
        }  # optional

        if (length_stat):
            max_input_len, max_pred_len, median_len = get_seq_length(args, total_dataset)
            data_objects["max_input_len"] = max_input_len.item()
            data_objects["max_pred_len"] = max_pred_len.item()
            data_objects["median_len"] = median_len.item()
            print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

        return data_objects

    ##################################################################
    ### USHCN dataset ###
    elif dataset_name == "ushcn":
        args.n_months = 48  # 48 monthes
        args.pred_window = 1  # predict future one month

        ### list of tuples (record_id, tt, vals, mask) ###
        total_dataset = USHCN('../data/ushcn/', n_samples=args.n, device=device)

        # Shuffle and split
        seen_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, random_state=42,
                                                                shuffle=True)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.75, random_state=42,
                                                                shuffle=False)
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
        test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
        print("Test record ids (first 20):", test_record_ids[:20])
        print("Test record ids (last 20):", test_record_ids[-20:])

        record_id, tt, vals, mask = train_data[0]

        input_dim = vals.size(-1)

        data_min, data_max, time_max = get_data_min_max(seen_data, device)  # (n_dim,), (n_dim,)

        if (patch_ts):
            collate_fn = USHCN_patch_variable_time_collate_fn
        else:
            collate_fn = USHCN_variable_time_collate_fn

        train_data = USHCN_time_chunk(train_data, args, device)
        val_data = USHCN_time_chunk(val_data, args, device)
        test_data = USHCN_time_chunk(test_data, args, device)
        batch_size = args.batch_size
        print("Dataset n_samples after time split:", len(train_data) + len(val_data) + len(test_data), \
              len(train_data), len(val_data), len(test_data))
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max))
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                    collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max))

        data_objects = {
            "train_dataloader": utils.inf_generator(train_dataloader),
            "val_dataloader": utils.inf_generator(val_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "input_dim": input_dim,
            "n_train_batches": len(train_dataloader),
            "n_val_batches": len(val_dataloader),
            "n_test_batches": len(test_dataloader),
            # "attr": total_dataset.params, #optional
            "data_max": data_max,  # optional
            "data_min": data_min,
            "time_max": time_max
        }  # optional

        if (length_stat):
            max_input_len, max_pred_len, median_len = USHCN_get_seq_length(args, train_data + val_data + test_data)
            data_objects["max_input_len"] = max_input_len.item()
            data_objects["max_pred_len"] = max_pred_len.item()
            data_objects["median_len"] = median_len.item()
            # data_objects["batch_size"] = args.batch_size * (args.n_months - args.pred_window + 1 - args.history)
            print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

        return data_objects


    ##################################################################
    ### Activity dataset ###
    elif dataset_name == "activity":
        args.pred_window = 1000  # predict future 1000 ms

        total_dataset = PersonActivity('../data/activity/', n_samples=args.n, download=True, device=device)

        # Shuffle and split
        seen_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, random_state=42,
                                                                shuffle=True)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.75, random_state=42,
                                                                shuffle=False)
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
        test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
        print("Test record ids (first 20):", test_record_ids[:20])
        print("Test record ids (last 20):", test_record_ids[-20:])

        record_id, tt, vals, mask = train_data[0]

        input_dim = vals.size(-1)

        batch_size = min(min(len(seen_data), args.batch_size), args.n)
        data_min, data_max, _ = get_data_min_max(seen_data, device)  # (n_dim,), (n_dim,)
        time_max = torch.tensor(args.history + args.pred_window)
        print('manual set time_max:', time_max)

        if (patch_ts):
            collate_fn = patch_variable_time_collate_fn
        else:
            collate_fn = variable_time_collate_fn

        train_data = Activity_time_chunk(train_data, args, device)
        val_data = Activity_time_chunk(val_data, args, device)
        test_data = Activity_time_chunk(test_data, args, device)
        batch_size = args.batch_size
        print("Dataset n_samples after time split:", len(train_data) + len(val_data) + len(test_data), \
              len(train_data), len(val_data), len(test_data))
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda batch: collate_fn(batch, args, device, data_type="train",
                                                                          data_min=data_min, data_max=data_max,
                                                                          time_max=time_max))
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                    collate_fn=lambda batch: collate_fn(batch, args, device, data_type="val",
                                                                        data_min=data_min, data_max=data_max,
                                                                        time_max=time_max))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: collate_fn(batch, args, device, data_type="test",
                                                                         data_min=data_min, data_max=data_max,
                                                                         time_max=time_max))

        data_objects = {
            "train_dataloader": utils.inf_generator(train_dataloader),
            "val_dataloader": utils.inf_generator(val_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "input_dim": input_dim,
            "n_train_batches": len(train_dataloader),
            "n_val_batches": len(val_dataloader),
            "n_test_batches": len(test_dataloader),
            # "attr": total_dataset.params, #optional
            "data_max": data_max,  # optional
            "data_min": data_min,
            "time_max": time_max
        }  # optional

        if (length_stat):
            max_input_len, max_pred_len, median_len = Activity_get_seq_length(args, train_data + val_data + test_data)
            data_objects["max_input_len"] = max_input_len.item()
            data_objects["max_pred_len"] = max_pred_len.item()
            data_objects["median_len"] = median_len.item()
            print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

        return data_objects
