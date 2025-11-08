from data_provider.data_loader import Dataset_P12, Dataset_MIMIC, Dataset_activity
import torch
from torch.utils.data import DataLoader

data_dict = {
    'P12': Dataset_P12,
    'MIMIC': Dataset_MIMIC,
    'activity': Dataset_activity,
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    if args.task_name == 'classification':
        if args.data == 'P12' or args.data == 'P19' or args.data == 'PAM' or args.data == 'MIMIC' or args.data == 'activity':
            data_set = Data(args=args, dataset=args.data, device=torch.device("cpu"), q=args.quantization, upsampling_batch=False)
            return data_set, None