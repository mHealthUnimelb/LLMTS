# q: Quantization factor for time steps. flag: A control flag that changes how the function returns the data loaders.
def get_physionet_data(args, device, q, flag=1):
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
    #
    # # Combine and shuffle samples from physionet Train and physionet Test
    # total_dataset = train_dataset_obj[:len(train_dataset_obj)]
    #
    # if not args.classif:
    #     # Concatenate samples from original Train and Test sets
    #     # Only 'training' physionet samples are have labels.
    #     # Therefore, if we do classifiction task, we don't need physionet 'test' samples.
    #     total_dataset = total_dataset + \
    #                     test_dataset_obj[:len(test_dataset_obj)]

    total_dataset = PhysioNet('data/physionet',
                              quantization=q,
                              download=True,
                              device=device)

    print(len(total_dataset))  # 4000
    # Shuffle and split total_dataset into train/test sets
    train_data, temp_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                             random_state=42, shuffle=True)
    if args.classif:
        # if classification task, we further split into train and validation sets
        val_data, test_data = model_selection.train_test_split(temp_data, train_size=0.5,
                                                                 random_state=42, shuffle=False)

    # tt: time steps, vals: observed values, mask: which values are observed
    record_id, tt, vals, mask, labels = train_data[0]

    # n_samples = len(total_dataset)
    input_dim = vals.size(-1) # determine the number of features. vals: [T, D], where D is the number of features
    data_min, data_max = get_data_min_max(total_dataset, device) # Compute the minimum and maximum values across all features in the entire dataset
    batch_size = min(len(train_data), args.batch_size) # ensures the batch size isn't larger than the dataset or user-specified number
    if flag:
        # combines variable-length time series into a single tensor, normalizing and preparing them for model input
        test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif,
                                                      data_min=data_min, data_max=data_max)

        if args.classif:
            # if classification task, we further split the training data into train and validation sets
            # train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
            #                                                         random_state=11, shuffle=True)
            # collate training and validation sets, variable_time_collate_fn returns (data, labels)
            train_data_combined = variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            val_data_combined = variable_time_collate_fn(
                val_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            print(train_data_combined[1].sum(
            ), val_data_combined[1].sum(), test_data_combined[1].sum())  # tensor(356.) tensor(91.) tensor(107.)
            print(train_data_combined[0].size(), train_data_combined[1].size(),
                  val_data_combined[0].size(), val_data_combined[1].size(),
                  test_data_combined[0].size(), test_data_combined[1].size())
            # torch.Size([2560, 190, 83]) torch.Size([2560, 1]) torch.Size([640, 186, 83]) torch.Size([640, 1]) torch.Size([800, 203, 83]) torch.Size([800, 1])

            # convert the combined data (a tuple of data and labels) into TensorDatasets
            train_data_combined = TensorDataset(
                train_data_combined[0], train_data_combined[1].long().squeeze())
            val_data_combined = TensorDataset(
                val_data_combined[0], val_data_combined[1].long().squeeze())
            test_data_combined = TensorDataset(
                test_data_combined[0], test_data_combined[1].long().squeeze())
        else:
            # if not classification (e.g., regression/forecasting)
            train_data_combined = variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            print(train_data_combined.size(), test_data_combined.size())

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

    attr_names = total_dataset.params # get attribute names (parameter names) from training dataset object
    data_objects = {"dataset_obj": total_dataset,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": input_dim, # number of features
                    "n_train_batches": len(train_dataloader), # number of batches in train
                    "n_test_batches": len(test_dataloader),
                    "attr": attr_names,  # optional
                    "classif_per_tp": False,  # (optional) boolean flag indicating classification per time point or not
                    "n_labels": 1}  # (optional) how many labels per sample are expected
    if args.classif:
        # if classification, also create and store a validation DataLoader
        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
        data_objects["val_dataloader"] = val_dataloader
    return data_objects # return all the prepared data and metadata as a dictionary