from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, Dataset_ECG, Dataset_Physionet, Dataset_P12, Dataset_MIMIC, Dataset_activity
import torch
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'ECG': Dataset_ECG,
    'PhysioNet': Dataset_Physionet,
    'P12': Dataset_P12,
    'MIMIC': Dataset_MIMIC,
    'activity': Dataset_activity,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.task_name == 'classification':
        if args.data == 'P12' or args.data == 'P19' or args.data == 'PAM' or args.data == 'MIMIC' or args.data == 'activity':
            data_set = Data(args=args, dataset=args.data, device=torch.device("cpu"), q=args.quantization,
                            upsampling_batch=False)
            # data_loader =
            # if flag == 'train':
            #     data_loader = data_set.data_objects["train_dataloader"]
            # elif flag == 'val':
            #     data_loader = data_set.data_objects["val_dataloader"]
            # elif flag == 'test':
            #     data_loader = data_set.data_objects["test_dataloader"]
            return data_set, None
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            seq_len=args.seq_len,
            training_flag=args.is_training
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns
            )
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                percent=percent,
                seasonal_patterns=args.seasonal_patterns
            )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader