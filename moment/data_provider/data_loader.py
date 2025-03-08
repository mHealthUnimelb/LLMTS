import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import warnings
from data_provider.physionet import PhysioNet, get_data_min_max, variable_time_collate_fn2
from sklearn import model_selection
from data_provider.utils import *

warnings.filterwarnings('ignore')


class Dataset_Physionet(Dataset):
    def __init__(self, root_path=None, data_path=None, args=None, device='cpu', dataset_flag=1, flag='train', q=0, seq_len=2500, training_flag=1):
        print("batch size: ", args.batch_size)
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

        # n_samples = len(total_dataset)
        input_dim = vals.size(-1)
        data_min, data_max = get_data_min_max(total_dataset, device)
        batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
        if dataset_flag:
            test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif,
                                                          data_min=data_min, data_max=data_max)

            if args.classif:
                train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                                        random_state=11, shuffle=True)
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

        attr_names = train_dataset_obj.params
        data_objects = {"dataset_obj": train_dataset_obj,
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







