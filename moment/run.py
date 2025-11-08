import argparse
import os
import torch

from exp.exp_ir_classification import Exp_IR_Classification

import random
import numpy as np

parser = argparse.ArgumentParser(description='Moment')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--num_variables', type=int, default=7, help='number of variable')

parser.add_argument('--root_path', type=str, default='./data/raw_data/ETTh1/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--data_split_path', type=str)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--quantization', type=float, default=0.1,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--classify-pertp', action='store_true')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# classification task
parser.add_argument('--num_classes', type=int, default=2, help='number of class')

# optimization
parser.add_argument('--num_workers', type=int, default=90, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')
# parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--decay_fac', type=float, default=0.75)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

print('Args in experiment:')
print(args)

if args.task_name == 'classification':
    Exp = Exp_IR_Classification

if args.is_training:
    mses = []
    maes = []
    smapes = []
    msaes = []
    owas = []
    mapes = []
    accuraies = []
    auprcs = []
    aucs = []
    precisions = []
    recalls = []
    f1s = []

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_{}'.format(
            args.task_name,
            args.data,
            args.data_split_path.split("/")[-1].rsplit(".", 1)[0],
            args.seq_len,
            ii)

        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        exp = Exp(args)  # set experiments

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        best_model_path = path + '/' + 'checkpoint.pth'
        exp.model.load_state_dict(torch.load(best_model_path))

        if args.task_name == 'long_term_forecast':
            mse, mae = exp.test(setting)
            mses.append(mse)
            maes.append(mae)
        elif args.task_name == 'classification':
            if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
                accuracy, auprc, auc = exp.test(setting)
                accuraies.append(accuracy)
                auprcs.append(auprc)
                aucs.append(auc)
            elif args.data == 'PAM' or args.data == 'activity':
                accuracy, auprc, auc, precision, recall, F1 = exp.test(setting)
                accuraies.append(accuracy)
                auprcs.append(auprc)
                aucs.append(auc)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(F1)
        torch.cuda.empty_cache()

    if args.task_name == 'long_term_forecast':
        print('mse_means: ', np.array(mses), 'mean: ', np.mean(np.array(mses)))
        print('mae_means: ', np.array(maes), 'mean: ', np.mean(np.array(maes)))
    elif args.task_name == 'classification':
        if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
            print('accuracy:', np.mean(np.array(accuraies)))
            print('auprc:', np.mean(np.array(auprcs)))
            print('auc:', np.mean(np.array(aucs)))
        elif args.data == 'PAM' or args.data == 'activity':
            print('accuracy:', np.mean(np.array(accuraies)))
            print('auprc:', np.mean(np.array(auprcs)))
            print('auc:', np.mean(np.array(aucs)))
            print('precision:', np.mean(np.array(precisions)))
            print('recall:', np.mean(np.array(recalls)))
            print('F1 score:', np.mean(np.array(f1s)))

else:
    ii = 0
    setting = '{}_{}_{}_{}_{}'.format(
        args.task_name,
        args.data,
        args.data_split_path.split("/")[-1].rsplit(".", 1)[0],
        args.seq_len,
        ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    if args.task_name == 'long_term_forecast':
        mse, mae = exp.test(setting, test=1)
        print("mse:", mse)
        print("mae:", mae)
    elif args.task_name == 'classification':
        if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
            accuracy, auprc, auc = exp.test(setting, test=1)
            print('accuracy:', accuracy)
            print('auprc:', auprc)
            print('auc:', auc)
        elif args.data == 'PAM' or args.data == 'activity':
            accuracy, auprc, auc, precision, recall, F1 = exp.test(setting, test=1)
            print('accuracy:', accuracy)
            print('auprc:', auprc)
            print('auc:', auc)
            print('precision:', precision)
            print('recall:', recall)
            print('F1 score:', F1)
    torch.cuda.empty_cache()
