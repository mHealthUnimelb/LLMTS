import argparse
import numpy as np
import pathlib
import torch
import torch.nn as nn
import random

from config import Config
from data_loader_full import load_forecasting_data
from aeon.datasets import write_to_ts_file

parser = argparse.ArgumentParser(description="Time Series")
parser.add_argument("--batch_size", type=int, help="batch_size")
parser.add_argument("--seed", type=int, help="random seed")
parser.add_argument("--problemname", required=True, type=str, help="dataset problemname")
parser.add_argument("--out_dir", default="ts_outputs", type=str, help="Folder for generated .ts file")
parser.add_argument("--out_filename", required=True, type=str, help="file name")

args = parser.parse_args()

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
config = Config(args)

data_obj = load_forecasting_data(config)
train_loader = data_obj["train_dataloader"]
val_loader = data_obj["val_dataloader"]
test_loader = data_obj["test_dataloader"]

########## Train data ###########
full_data = []

for batch_idx, batch in enumerate(train_loader):
    # Get data
    seen_data, seen_tp, seen_mask = batch['observed_data'], batch['observed_tp'], batch['observed_mask'],
    future_data, future_tp, future_mask = batch['data_to_predict'], batch['tp_to_predict'], batch['mask_predicted_data']  
    # print('checking data shape', seen_data.shape, future_data.shape)
    ## prepare the data for moment model
    n_channels = seen_data.shape[2] 


    seen_data = seen_data.permute(0, 2, 1).to(device).float()
    future_data = future_data.permute(0, 2, 1).to(device).float()
    future_mask = future_mask.permute(0, 2, 1).to(device).float()

    # pad to 512 
    seen_data = torch.nn.functional.pad(seen_data, (0, 512 - seen_data.shape[2]), mode='constant', value=0)
    future_data = torch.nn.functional.pad(future_data, (0, 512 - future_data.shape[2]), mode='constant', value=0)
    future_mask = torch.nn.functional.pad(future_mask, (0, 512 - future_mask.shape[2]), mode='constant', value=0)

    # Store original dimensions
    batch_size, n_channels, seq_len = seen_data.shape

    # concat train and test
    full_seq = torch.cat([seen_data, future_data], dim=2)

    # full_data.append(full_seq.cpu().numpy())
    full_data.append(full_seq.cpu())

# full_data = np.concatenate(full_data, axis=0)
full_data = torch.cat(full_data, dim=0)

print(full_data.shape)

print("full data[0]", full_data[0])
print("full data[0]", full_data[1])

out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
outpath = out_dir / (args.out_filename+"_TRAIN.pt")

# write_to_ts_file(full_data, path=out_dir, problem_name="physionet_TRAIN_1", regression=True)
torch.save(full_data, outpath)

########## Test data ###########
full_data = []

for batch_idx, batch in enumerate(test_loader):
    # Get data
    seen_data, seen_tp, seen_mask = batch['observed_data'], batch['observed_tp'], batch['observed_mask'],
    future_data, future_tp, future_mask = batch['data_to_predict'], batch['tp_to_predict'], batch['mask_predicted_data']  
    # print('checking data shape', seen_data.shape, future_data.shape)
    ## prepare the data for moment model
    n_channels = seen_data.shape[2] 


    seen_data = seen_data.permute(0, 2, 1).to(device).float()
    future_data = future_data.permute(0, 2, 1).to(device).float()
    future_mask = future_mask.permute(0, 2, 1).to(device).float()

    # pad to 512 
    seen_data = torch.nn.functional.pad(seen_data, (0, 512 - seen_data.shape[2]), mode='constant', value=0)
    future_data = torch.nn.functional.pad(future_data, (0, 512 - future_data.shape[2]), mode='constant', value=0)
    future_mask = torch.nn.functional.pad(future_mask, (0, 512 - future_mask.shape[2]), mode='constant', value=0)

    # Store original dimensions
    batch_size, n_channels, seq_len = seen_data.shape

    # concat train and test
    full_seq = torch.cat([seen_data, future_data], dim=2)

    # full_data.append(full_seq.cpu().numpy())
    full_data.append(full_seq.cpu())

# full_data = np.concatenate(full_data, axis=0)
full_data = torch.cat(full_data, dim=0)

print(full_data.shape)

print("full data[0]", full_data[0])
print("full data[0]", full_data[1])

out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
outpath = out_dir / (args.out_filename+"_TEST.pt")

# write_to_ts_file(full_data, path=out_dir, problem_name="physionet_TEST_1", regression=True)
torch.save(full_data, outpath)