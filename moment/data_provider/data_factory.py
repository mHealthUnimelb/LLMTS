from data_provider.data_loader import Dataset_P12, Dataset_MIMIC
import torch
from torch.utils.data import DataLoader

data_dict = {
    'P12': Dataset_P12,
    'MIMIC': Dataset_MIMIC,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    # timeenc = 0 if args.embed != 'timeF' else 1
    # percent = args.percent

    if args.task_name == 'classification':
        if args.data == 'P12' or args.data == 'P19' or args.data == 'PAM' or args.data == 'MIMIC':
            data_set = Data(args=args, dataset=args.data, device=torch.device("cpu"), q=args.quantization, upsampling_batch=False)
            return data_set, None

    # if flag == 'test':
    #     shuffle_flag = False
    #     drop_last = True
    #     if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
    #         batch_size = args.batch_size
    #     else:
    #         batch_size = 1
    #     freq = args.freq
    # else:
    #     shuffle_flag = True
    #     drop_last = True
    #     batch_size = args.batch_size
    #     freq = args.freq
    #
    # if args.task_name == 'anomaly_detection':
    #     drop_last = False
    #     data_set = Data(
    #         root_path=args.root_path,
    #         win_size=args.seq_len,
    #         flag=flag,
    #     )
    #     print(flag, len(data_set))
    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last)
    #     return data_set, data_loader
    # elif args.task_name == 'classification':
    #     if args.data == 'P12' or args.data == 'P19' or args.data == 'PAM':
    #         data_set = Data(args=args, dataset=args.data, device=torch.device("cpu"), upsampling_batch=False)
    #         return data_set, None
    #     drop_last = False
    #     data_set = Data(
    #         root_path=args.root_path,
    #         flag=flag,
    #     )
    #     print(flag, len(data_set))
    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last,
    #         collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    #     )
    #     return data_set, data_loader
    # else:
    #     if args.data == 'm4':
    #         drop_last = False
    #     data_set = Data(
    #         root_path=args.root_path,
    #         data_path=args.data_path,
    #         flag=flag,
    #         size=[args.seq_len, args.label_len, args.pred_len],
    #         features=args.features,
    #         target=args.target,
    #         timeenc=timeenc,
    #         freq=freq,
    #         seasonal_patterns=args.seasonal_patterns,
    #         percent=percent
    #     )
    #     # batch_size = args.batch_size
    #     print(flag, len(data_set))
    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last)
    #     return data_set, data_loader
