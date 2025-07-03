from data_processing.physionet_full import PhysioNet_full, get_data_min_max_inter
from data_processing.physionet_classification import PhysioNet_classify, get_data_min_max
from data_processing.person_activity import PersonActivity, Activity_time_chunk
from data_processing.mimic import MIMIC
from sklearn import model_selection
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import sys
import time

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
    D = batch[0][2].shape[1]  # Number of features
    N = batch[0][-1].shape[0] if activity else 1  # Number of labels
    ## because for each_id, they show different length of time series, then here we want to make sure 
    # all batches have the same length of time series, then we padding to ensure the shorter sequences are extended
    # to match the longest sequences in the batch
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = max(len_tt)

    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)

    if not activity:
        for b, (record_id, tt, vals, mask) in enumerate(batch):
            currlen = tt.size(0)
            enc_combined_tt[b, :currlen] = tt.to(device)
            enc_combined_vals[b, :currlen] = vals.to(device)
            enc_combined_mask[b, :currlen] = mask.to(device)
    else:
        for b, (record_id, tt, vals, mask, _) in enumerate(batch):
            currlen = tt.size(0)
            enc_combined_tt[b, :currlen] = tt.to(device)
            enc_combined_vals[b, :currlen] = vals.to(device)
            enc_combined_mask[b, :currlen] = mask.to(device)


    # Normalize data
    if not activity:
        enc_combined_vals = normalize_masked_data(enc_combined_vals, enc_combined_mask, data_min, data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    data_dict = {
        "data": enc_combined_vals,  # (batch_size, maxlen, D)
        "time_steps": enc_combined_tt,  # (batch_size, maxlen)
        "mask": enc_combined_mask,  # (batch_size, maxlen, D)
    }


    return data_dict


# def normalize_masked_data(data, mask, att_min, att_max):
#     """
#     Normalize data using min-max scaling with masking.
#     """
#     data = (data - att_min) / (att_max - att_min + 1e-6)
#     data = data * mask
#     return data

def normalize_masked_data(data, mask, att_min, att_max):
	scale = att_max - att_min
	scale = scale + (scale == 0) * 1e-8
	# we don't want to divide by zero
	if (scale != 0.).all(): 
		# Ensure tensors are on the same device
		att_min = att_min.to(data.device)
		scale = scale.to(data.device)
		data_norm = (data - att_min) / scale
	else:
		raise Exception("Zero!")

	# set masked out elements back to zero 
	data_norm[mask == 0] = 0

	if torch.isnan(data_norm).any():
		raise Exception("nans!")

	return data_norm

def normalize_masked_tp(data, att_min, att_max):
    # Convert scalar values to tensors on the correct device if needed
    if not isinstance(att_min, torch.Tensor):
        att_min = torch.tensor(att_min, device=data.device)
    if not isinstance(att_max, torch.Tensor):
        att_max = torch.tensor(att_max, device=data.device)
        
    scale = att_max - att_min
    scale = (scale + (scale == 0) * 1e-8).to(data.device)  # Avoid division by zero
    # we don't want to divide by zero
    if (scale != 0.).all():
        # print('check device', data.device, att_min.device, scale.device)
        data_norm = (data - att_min) / scale
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm

def variable_time_collate_fn_fore(batch, config, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None, time_max = None):
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
		n_observed_tp = torch.lt(tt, config.history).sum()
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


	observed_data = normalize_masked_data(observed_data, observed_mask, 
			att_min = data_min, att_max = data_max)
	predicted_data = normalize_masked_data(predicted_data, predicted_mask, 
			att_min = data_min, att_max = data_max)
	
	observed_tp = normalize_masked_tp(observed_tp, att_min = 0, att_max = time_max)
	predicted_tp = normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
		
	data_dict = {"observed_data": observed_data,
			"observed_tp": observed_tp,
			"observed_mask": observed_mask,
			"data_to_predict": predicted_data,
			"tp_to_predict": predicted_tp,
			"mask_predicted_data": predicted_mask,
			}
	
	return data_dict

def get_data_min_max(records, device):
	inf = torch.Tensor([float("Inf")])[0].to(device)

	data_min, data_max, time_max = None, None, -inf

	for b, (record_id, tt, vals, mask) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

		time_max = torch.max(time_max, tt.max())

	print('data_max:', data_max)
	print('data_min:', data_min)
	print('time_max:', time_max)

	return data_min, data_max, time_max

def load_forecasting_data(config):
    """
    Load the data based on the configuration provided.
    """
    device = config.device
    if config.dataset_name in ["physionet", "mimic"]:
        if config.dataset_name == "physionet":
            total_dataset = PhysioNet_full(
                'data/forecast/physionet', ## modify the path here
                quantization=config.quantization,
                download=True,
                n_samples=config.n,
                device=device
            )
        elif config.dataset_name == "mimic":
             total_dataset = MIMIC('data/forecast/mimic', n_samples = config.n, device = device)

        # only use subset of 1000 samples
        
        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=config.seed, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=config.seed, shuffle=False
        )
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))

    

        # Get data min, max, and time max for normalization
        data_min, data_max, time_max = get_data_min_max_inter(train_data, device)
        # Make sure data_min and data_max are on the same device as the rest of the data
        data_min = data_min.to(device)
        data_max = data_max.to(device)
        time_max = time_max.to(device)

        # Create data loaders to sperate data based on their time length
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, config, device, data_type="train",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, config, device, data_type="test",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, config, device, data_type="val",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        
        data_obj = {
					"train_dataloader": train_dataloader, 
					"val_dataloader": val_dataloader,
					"test_dataloader": test_dataloader,
					"input_dim": train_data[0][2].shape[1],
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					# "attr": total_dataset.params, #optional
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max
					} #optional
    elif config.dataset_name == "activity":
        config.pred_window = 1000 # predict future 1000 ms
        # Load PhysioNet dataset
        total_dataset = PersonActivity(
            'data/forecast/activity',
            download=True,
            n_samples=config.n,
            device=device
        )
        # total_dataset = torch.load('/home/yw573/rds/hpc-work/nips25/baselines/t-PatchGNN/data/activity/processed/data.pt')

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=config.seed, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=config.seed, shuffle=False
        )
        print("whole samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
        test_record_ids = [record_id for record_id, tt, vals, mask in test_data]


        # Get data min, max, and time max for normalization
        data_min, data_max, _= get_data_min_max(total_dataset, device)
        time_max = torch.tensor(config.history + config.pred_window)
        # print('manual set time_max:', time_max)

        train_data = Activity_time_chunk(train_data, config, device)
        val_data = Activity_time_chunk(val_data, config, device)
        test_data = Activity_time_chunk(test_data, config, device)
        print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
        
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, config, device, data_type="train",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, config, device, data_type="val",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn_fore(batch, config, device, data_type="test",
                                                              data_min=data_min, data_max=data_max, time_max=time_max)
        )
        
        data_obj = {
					"train_dataloader": train_dataloader, 
					"val_dataloader": val_dataloader,
					"test_dataloader": test_dataloader,
					"input_dim": train_data[0][2].shape[1],
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					# "attr": total_dataset.params, #optional
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max
					} #optional


    return data_obj
    

def load_interpolation_data(config):
    """
    Load the data based on the configuration provided.
    """
    device = config.device
    #/home/yw573/rds/hpc-work/irregular/t-PatchGNN/data/physionet_label
    if config.dataset_name == "physionet":
        # Load PhysioNet dataset
        total_dataset = PhysioNet_full(
            'data/interpolation/physionet',
            quantization=config.quantization,
            download=True,
            n_samples=config.n,
            device=device
        )

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=config.seed, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=config.seed, shuffle=False
        )
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))


        # Get data min, max, and time max for normalization
        data_min, data_max, time_max = get_data_min_max_inter(total_dataset, device)

        # Create data loaders
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False,
                                                              data_min=data_min, data_max=data_max)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False,
                                                              data_min=data_min, data_max=data_max)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False,
                                                              data_min=data_min, data_max=data_max)
        )

        data_obj = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader,
            "input_dim": train_data[0][2].shape[1],
            "data_min": data_min,
            "data_max": data_max,
            "time_max": time_max,
        }

    elif config.dataset_name == "activity":
        # Load PhysioNet dataset
        total_dataset = PersonActivity(
            'data/interpolation/activity',
            download=True,
            n_samples=config.n,
            device=device
        )

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=config.seed, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=config.seed, shuffle=False
        )
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))


        # Get data min, max, and time max for normalization
        data_min, data_max, _= get_data_min_max(total_dataset, device)

        # Create data loaders
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, activity=True,
                                                              data_min=data_min, data_max=data_max)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, activity=True,
                                                              data_min=data_min, data_max=data_max)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, activity=True,
                                                              data_min=data_min, data_max=data_max)
        )

        data_obj = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader,
            "input_dim": train_data[0][2].shape[1],
            "data_min": data_min,
            "data_max": data_max
        }
    return data_obj



def load_classification_data(config):
    """
    Load data specifically for PhysioNet mortality prediction classification task
    """
    device = config.device
    
    if config.dataset_name in ["physionet", "mimic"]:
        if config.dataset_name == "physionet":
        # Load PhysioNet dataset
            total_dataset = PhysioNet_classify(
                '/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/physionet_classification', ##### change here to your path
                quantization=config.quantization,
                download=True,  # Data already exists
                n_samples=config.n,
                device=device
            )
        elif config.dataset_name == "mimic":
            total_dataset = MIMIC('/home/yw573/rds/hpc-work/irregular/t-PatchGNN/data/mimic_classification', n_samples = config.n, device = device)
        
        # Use subset if specified (for faster testing)
        # if hasattr(config, 'use_subset') and config.use_subset:
        #     subset_size = min(1000, len(total_dataset))
        #     total_dataset = total_dataset[:subset_size]
        
        print("Dataset size:", len(total_dataset))
        
        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=42, shuffle=False
        )
        print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Get data min, max, and time max for normalization
        data_min, data_max, time_max = get_data_min_max(total_dataset, device)
        
        # Create data loaders with classification-specific collate function
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max)
        )
        data_obj = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader,
            "input_dim": train_data[0][2].shape[1],
            "data_min": data_min,
            "data_max": data_max,
            "time_max": time_max,
             
        }
        
    elif config.dataset_name == "activity":
        total_dataset = PersonActivity('/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/activity_classification',
                                 download=True, n_samples=config.n, device=device)
          
          # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            train_data, train_size=0.75, random_state=42, shuffle=False
        )
        print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Get data min, max, and time max for normalization
        data_min, data_max, time_max = get_data_min_max(total_dataset, device)
        
        # Create data loaders with classification-specific collate function
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max,activity = True)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max,activity = True)
        )
        test_dataloader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_classification(batch, device, data_min, data_max,activity = True)
        )

        data_obj = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader,
            "input_dim": train_data[0][2].shape[1],
            "data_min": data_min,
            "data_max": data_max,
            "time_max": time_max,
             
        }
        
    return data_obj

def variable_time_collate_classification(batch, device=torch.device("cpu"), data_min=None, data_max=None,activity = False):
    """
    Collate function for classification tasks with PhysioNet mortality labels
    """
    # Remove the debugging code that's causing sys.exit()
    D = batch[0][2].shape[1]  # Number of features
    N = batch[0][-1].shape[1] if activity else 1
    # Get max sequence length in the batch
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = max(len_tt)
    
    # Initialize tensors with zeros
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    if activity:
        combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
    else:
        combined_labels = torch.zeros([len(batch)], dtype=torch.long).to(device)
    
    # Fill tensors with data
    # can we still save the record_id to the batch

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
 
        # Handle mortality label (binary classification)
        if labels is not None:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:   
                # combined_labels[b] = labels.long().to(device)  # Convert to long for CrossEntropyLoss
                combined_labels[b] = torch.tensor(labels, dtype=torch.long).to(device)
        
    
    # Normalize data
    if not activity:
        enc_combined_vals = normalize_masked_data(enc_combined_vals, enc_combined_mask, data_min, data_max)
    
    # Normalize time steps
    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)
    
    return {
        "data": enc_combined_vals,
        "time_steps": enc_combined_tt,
        "mask": enc_combined_mask,
        "labels": combined_labels,
        # "record_id": record_id

    }

def load_data_old(config):
    """
    Load the data based on the configuration provided.
    """
    device = config.device

    if config.dataset_name in ["physionet", "mimic"]:
        # Load PhysioNet dataset
        physio_dataset = PhysioNet_full(
            '/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/physionet_full',
            quantization=config.quantization,
            download=True,
            n_samples=config.n,
            device=device
        )

        mimic_dataset = MIMIC('/home/yw573/rds/hpc-work/irregular/t-PatchGNN/data/mimic_old',
                              n_samples=config.n, device=device)
        
        # combine datasets
        total_dataset = physio_dataset[:len(physio_dataset)] + mimic_dataset[:len(mimic_dataset)]
        print(len(total_dataset))

        # Shuffle and split
        train_data, val_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data))


        # Get data min, max, and time max for normalization
        # we now combine two datasets, so we have totally 41 + 96 variables
        data_min, data_max, time_max = get_data_min_max_inter(total_dataset, device)

        # Create data loaders
        train_dataloader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True, drop_last=True,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False,
                                                              data_min=data_min, data_max=data_max)
        )
        val_dataloader = DataLoader(
            val_data, batch_size=config.batch_size, shuffle=False, drop_last=True,
            collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False,
                                                              data_min=data_min, data_max=data_max)
        )
       
        data_obj = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "input_dim": train_data[0][2].shape[1],
            "data_min": data_min,
            "data_max": data_max,
            "time_max": time_max,
        }

    
    return data_obj

class MultiDatasetIterator:
    """Iterator that alternates between multiple datasets"""
    def __init__(self, dataloaders, cycle_mode=False):
        self.dataloaders = dataloaders
        self.iterators = [iter(dl) for dl in dataloaders]
        self.dataset_idx = 0  # Track which dataset we're using
        self.exhausted_iterators = set()  # Track which iterators are exhausted
        self.cycle_mode = cycle_mode  # Whether to cycle indefinitely
        
    def __iter__(self):
        self.iterators = [iter(dl) for dl in self.dataloaders]
        self.dataset_idx = 0
        self.exhausted_iterators = set()
        return self
        
    def __next__(self):
        start_time = time.time()
        
        # Rest of your existing __next__ method...
        try:
            # Timeout check to prevent infinite loops
            if time.time() - start_time > 10:  # 10 second timeout
                print(f"Warning: Dataset iteration timeout for dataset {self.dataset_idx}")
                self.dataset_idx = (self.dataset_idx + 1) % len(self.dataloaders)
                return self.__next__()
                
            # Your existing code
            batch = next(self.iterators[self.dataset_idx])
            batch["dataset_idx"] = self.dataset_idx
            return batch
        except StopIteration:
            # Mark this iterator as exhausted
            self.exhausted_iterators.add(self.dataset_idx)
            
            if self.cycle_mode:
                # Reset this iterator
                self.iterators[self.dataset_idx] = iter(self.dataloaders[self.dataset_idx])
                
            # Move to next dataset
            self.dataset_idx = (self.dataset_idx + 1) % len(self.dataloaders)
            
            # If we've checked all datasets and they're all exhausted, stop
            if len(self.exhausted_iterators) == len(self.dataloaders):
                if not self.cycle_mode:
                    raise StopIteration
                else:
                    # Reset for next cycle
                    self.exhausted_iterators = set()
                
            # Try the next dataset
            return self.__next__()

    def __len__(self):
        """Return combined length of all dataloaders"""
        return sum(len(dl) for dl in self.dataloaders)

def load_data(config):
    """Load each dataset separately but combine them for training"""
    device = torch.device('cpu')
    datasets_info = []
    train_dataloaders = []
    val_dataloaders = []
    
    # Process PhysioNet dataset
    physio_dataset = PhysioNet_full(
        '/home/yw573/rds/hpc-work/nips25/baselines/train_from_scratch/data/physionet_full',
        quantization=config.quantization,
        download=True,
        n_samples=config.n,
        device=torch.device('cpu')
    )

    physio_dataset = physio_dataset[:32]  # Use a subset for faster testing
    
    # Split PhysioNet
    physio_train, physio_val = model_selection.train_test_split(
        physio_dataset, train_size=0.8, random_state=42, shuffle=True
    )
    
    # Get normalization parameters for PhysioNet
    physio_min, physio_max, physio_time_max = get_data_min_max_inter(physio_dataset, device)
    
    # Create PhysioNet dataloaders
    physio_train_loader = DataLoader(
        physio_train, batch_size=config.batch_size, shuffle=True,drop_last=True,
        collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, 
                                                         data_min=physio_min, data_max=physio_max)
    )
    physio_val_loader = DataLoader(
        physio_val, batch_size=config.batch_size, shuffle=False,drop_last=True,
        collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, 
                                                         data_min=physio_min, data_max=physio_max)
    )
    
    train_dataloaders.append(physio_train_loader)
    val_dataloaders.append(physio_val_loader)
    
    datasets_info.append({
        "name": "physionet",
        "input_dim": physio_train[0][2].shape[1],
        # "data_min": physio_min,
        # "data_max": physio_max,
        # "time_max": physio_time_max
    })
    
    # Process MIMIC dataset similarly...
    mimic_dataset = MIMIC(
        '/home/yw573/rds/hpc-work/irregular/t-PatchGNN/data/mimic_old',
        n_samples=config.n, device=torch.device('cpu')
    )

    mimic_dataset = mimic_dataset[:64]
    
    # Split MIMIC
    mimic_train, mimic_val = model_selection.train_test_split(
        mimic_dataset, train_size=0.8, random_state=42, shuffle=True
    )
    
    # Get normalization parameters for MIMIC
    mimic_min, mimic_max, mimic_time_max = get_data_min_max_inter(mimic_dataset, device)
    
    # Create MIMIC dataloaders
    mimic_train_loader = DataLoader(
        mimic_train, batch_size=config.batch_size, shuffle=True,drop_last=True,
        collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, 
                                                         data_min=mimic_min, data_max=mimic_max)
    )
    mimic_val_loader = DataLoader(
        mimic_val, batch_size=config.batch_size, shuffle=False,drop_last=True,
        collate_fn=lambda batch: variable_time_collate_fn(batch, device, classify=False, 
                                                         data_min=mimic_min, data_max=mimic_max)
    )
    
    train_dataloaders.append(mimic_train_loader)
    val_dataloaders.append(mimic_val_loader)
    
    datasets_info.append({
        "name": "mimic",
        "input_dim": mimic_train[0][2].shape[1],
        # "data_min": mimic_min,
        # "data_max": mimic_max,
        # "time_max": mimic_time_max
    })
    
    # Create combined iterators
    multi_train_iterator = MultiDatasetIterator(train_dataloaders)
    multi_val_iterator = MultiDatasetIterator(val_dataloaders)
    
    return {
        "train_dataloader": multi_train_iterator,
        "val_dataloader": multi_val_iterator,
        "datasets_info": datasets_info
    }