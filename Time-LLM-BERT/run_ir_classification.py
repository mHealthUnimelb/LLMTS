import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torchmetrics.classification import MulticlassAveragePrecision
from sklearn.metrics import confusion_matrix, average_precision_score, ConfusionMatrixDisplay, precision_recall_curve, \
    auc, roc_auc_score, precision_score, recall_score, f1_score

from models import Autoformer, DLinear, TimeLLM, TimeLLM_4, TimeLLM_5

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['CURL_CA_BUNDLE'] = ''
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content
from utils.tools_without_mark_ir import del_files, EarlyStopping, adjust_learning_rate, vali, test, load_content, cal_accuracy, one_hot

parser = argparse.ArgumentParser(description='Time-LLM')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--data_split_path', type=str, help='data split path')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss")
parser.add_argument('--classify-pertp', action='store_true')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# classificaton task
parser.add_argument('--num_classes', type=int, required=True, default=4, help='number of classes')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')  # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096',
                    help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()
fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

train_accuracies = []
vali_accuracies = []
test_accuracies = []
train_losses = []
vali_losses = []
test_losses = []
train_auprcs = []
vali_auprcs = []
test_auprcs = []

# auprc_metric = MulticlassAveragePrecision(num_classes=args.num_classes, average="macro")

# def auprc_metric(num_class, probs, target):
#     probs = probs.detach().cpu().numpy()
#     target = target.detach().cpu().numpy()
#
#     # Initialize list to store AUPRC for each class
#     auprcs = []
#
#     # Compute AUPRC for each class
#     for i in range(num_class):
#         # For class `i`, the true labels are `1` if the actual label is `i`, else `0`
#         precision, recall, _ = precision_recall_curve(target == i, probs[:, i])
#         auprc = auc(recall, precision)
#         auprcs.append(auprc)
#
#     # Return the average AUPRC across all classes
#     return np.mean(auprcs)

all_data, _ = data_provider(args, None)
train_loader = all_data.data_objects["train_dataloader"]
vali_loader = all_data.data_objects["val_dataloader"]
test_loader = all_data.data_objects["test_dataloader"]
dim = all_data.data_objects["input_dim"]

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_nc{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des,
            args.num_classes, ii)

        print("Initial memory allocated:", torch.cuda.memory_allocated() / (1024 ** 3))
        print("Initial memory reserved:", torch.cuda.memory_reserved() / (1024 ** 3))

        # train_data, train_loader = data_provider(args, 'train')
        # vali_data, vali_loader = data_provider(args, 'val')
        # test_data, test_loader = data_provider(args, 'test')

        print("data_provider memory allocated:", torch.cuda.memory_allocated() / (1024 ** 3))
        print("data_provider memory reserved:", torch.cuda.memory_reserved() / (1024 ** 3))

        args.content = load_content(args)
        print("prompt", args.content)

        if args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        elif args.model == 'TimeLLM_4':
            model = TimeLLM_4.Model(args).float()
        elif args.model == 'TimeLLM_5':
            model = TimeLLM_5.Model(args).float()
        else:
            model = TimeLLM.Model(args).float()

        print("model memory allocated:", torch.cuda.memory_allocated() / (1024 ** 3))
        print("model memory reserved:", torch.cuda.memory_reserved() / (1024 ** 3))

        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path

        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)  # The number of data points in each batch
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = nn.CrossEntropyLoss()
        # mae_metric = nn.L1Loss()
        metric = "accuracy"

        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

        print("data prepare memory allocated:", torch.cuda.memory_allocated() / (1024 ** 3))
        print("data prepare memory reserved:", torch.cuda.memory_reserved() / (1024 ** 3))

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            train_trues = []
            train_preds = []

            model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
                print("batch_x shape: ", batch_x.shape)
                print("batch_y shape: ", batch_y.shape)

                iter_count += 1
                model_optim.zero_grad()
                observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:, :, -1]

                batch_x = observed_data.float().to(accelerator.device)
                batch_y = batch_y.squeeze().long().to(accelerator.device)
                # observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:, :, -1]

                print("batch_x shape: ", batch_x.shape)  # [24, 2881, 83]
                print("batch_y shape: ", batch_y.shape)  # [24]

                print("train batch data memory allocated:", torch.cuda.memory_allocated() / (1024 ** 3))
                print("train batch data memory reserved:", torch.cuda.memory_reserved() / (1024 ** 3))

                # # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                #     accelerator.device)
                # dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                #     accelerator.device)

                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]["aligned_logits"]
                            # outputs = model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                            outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                            # outputs = model((torch.cat((observed_data, observed_mask), 2), observed_tp), x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                        else:
                            # outputs = model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)
                            outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
                            # outputs = model((torch.cat((observed_data, observed_mask), 2), observed_tp), x_mark_enc=None, x_dec=None, x_mark_dec=None)

                        print("model output memory allocated:", torch.cuda.memory_allocated() / (1024 ** 3))
                        print("model output memory reserved:", torch.cuda.memory_reserved() / (1024 ** 3))

                        # f_dim = -1 if args.features == 'MS' else 0
                        # outputs = outputs[:, -args.pred_len:, f_dim:]
                        # batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                        # loss = criterion(outputs, batch_y)
                        # train_loss.append(loss.item())
                else:
                    if args.output_attention:
                        # outputs = model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                        outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                        # outputs = model((torch.cat((observed_data, observed_mask), 2), observed_tp), x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                    else:
                        # outputs = model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)
                        outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
                        # outputs = model((torch.cat((observed_data, observed_mask), 2), observed_tp), x_mark_enc=None, x_dec=None, x_mark_dec=None)

                    print("model output memory allocated:", torch.cuda.memory_allocated() / (1024 ** 3))
                    print("model output memory reserved:", torch.cuda.memory_reserved() / (1024 ** 3))

                    # f_dim = -1 if args.features == 'MS' else 0
                    # outputs = outputs[:, -args.pred_len:, f_dim:]
                    # batch_y = batch_y[:, -args.pred_len:, f_dim:]
                print(f"outputs shape: {outputs.shape}, type: {outputs.dtype}")

                if args.classify_pertp:
                    outputs = outputs.reshape(-1, args.num_classes)
                    batch_y = batch_y.argmax(-1).reshape(-1)
            
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                train_trues.append(batch_y)
                train_preds.append(outputs)

                if (i + 1) % 100 == 0:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()

                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_preds = torch.cat(train_preds, 0)
            train_trues = torch.cat(train_trues, 0)
            train_probs = torch.nn.functional.softmax(train_preds)
            train_predictions = torch.argmax(train_probs, dim=1)
            train_trues = train_trues.flatten()
            # calculate accuracy
            train_accuracy = torch.mean((train_trues == train_predictions).float()).item()
            train_accuracies.append(train_accuracy)
            train_trues = train_trues.detach().cpu().numpy()
            if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
                # calculate AUROC
                train_auc = roc_auc_score(train_trues,
                                        train_probs.detach().cpu().float().numpy()[:,
                                        1]) if not args.classify_pertp else 0.

                # calculate AUPRC
                # train_auprc = auprc_metric(args.num_classes, train_probs, train_trues)
                # train_auprc = auprc_metric(train_probs, train_trues)
                train_auprc = average_precision_score(train_trues, train_probs.detach().cpu().float().numpy()[:,
                                                                1]) if not args.classify_pertp else 0.
                train_auprcs.append(train_auprc)

                train_loss = np.average(train_loss)
                train_losses.append(train_loss)
                print("before vali memory allocated:", torch.cuda.memory_allocated() / (1024 ** 3))
                print("before vali memory reserved:", torch.cuda.memory_reserved() / (1024 ** 3))
                vali_loss, vali_accuracy, vali_auc, vali_auprc = vali(args, accelerator, model, dim, vali_loader,
                                                                    criterion, metric)
                print("after vali memory allocated:", torch.cuda.memory_allocated() / (1024 ** 3))
                print("after vali memory reserved:", torch.cuda.memory_reserved() / (1024 ** 3))

                vali_losses.append(vali_loss)
                vali_accuracies.append(vali_accuracy)
                vali_auprcs.append(vali_auprc)
                test_loss, test_accuracy, test_auc, test_auprc = vali(args, accelerator, model, dim, test_loader,
                                                                    criterion, metric)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
                test_auprcs.append(test_auprc)
                accelerator.print(
                    "Epoch: {0} | Train Loss: {1:.7f} Train Acc: {2:.7f} Train AUROC: {3:.7f} Train AUPRC: {4:.7f} Vali Loss: {5:.7f} Vali Acc: {6:.7f} Vali AUROC: {7:.7f} Vali AUPRC: {8:.7f} Test Loss: {9:.7f} Test Acc: {10:.7f} Test AUROC: {11:.7f} Test AUPRC: {12:.7f}".format(
                        epoch + 1, train_loss, train_accuracy, train_auc, train_auprc, vali_loss, vali_accuracy, vali_auc,
                        vali_auprc, test_loss, test_accuracy, test_auc, test_auprc))
            elif args.data == 'PAM' or args.data == 'activity':
                train_auc = roc_auc_score(one_hot(train_trues),
                                          train_probs.detach().cpu().float().numpy())
                train_auprc = average_precision_score(one_hot(train_trues),
                                                      train_probs.detach().cpu().float().numpy())
                train_precision = precision_score(train_trues, train_probs.detach().cpu().float().numpy().argmax(1),
                                                  average='macro', )
                train_recall = recall_score(train_trues, train_probs.detach().cpu().float().numpy().argmax(1),
                                            average='macro', )
                train_F1 = 2 * (train_precision * train_recall) / (
                        train_precision + train_recall)
                
                train_auprcs.append(train_auprc)
                train_loss = np.average(train_loss)
                train_losses.append(train_loss)

                vali_loss, vali_accuracy, vali_auc, vali_auprc, vali_precision, vali_recall, vali_F1 = vali(args, accelerator, model, dim, vali_loader, criterion, metric)

                vali_losses.append(vali_loss)
                vali_accuracies.append(vali_accuracy)
                vali_auprcs.append(vali_auprc)

                test_loss, test_accuracy, test_auc, test_auprc, test_precision, test_recall, test_F1 = vali(args, accelerator, model, dim, test_loader, criterion, metric)
                
                # print(type(train_loss), train_loss)
                # print(type(train_auc), train_auc)
                # print(type(train_auprc), train_auprc)
                # print(type(train_precision), train_precision)
                # print(type(train_recall), train_recall)
                # print(type(train_F1), train_F1)
                # print(type(vali_loss), vali_loss)
                # print(type(vali_auc), vali_auc)
                # print(type(vali_auprc), vali_auprc)
                # print(type(vali_precision), vali_precision)
                # print(type(vali_recall), vali_recall)
                # print(type(vali_F1), vali_F1)

                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
                test_auprcs.append(test_auprc)
                accelerator.print("Epoch: {0} | Train Loss: {1:.7f} Train Acc: {2:.7f} Train AUPRC: {3:.7f} Train AUC: {4:.7f} Train Precision: {5:.7f} Train Recall: {6:.7f} Train F1 score: {7:.7f} Vali Loss: {8:.7f} Vali Acc: {9:.7f} Vali AUPRC: {10:.7f} Vali AUC: {11:.7f} Vali Precision: {12:.7f} Vali Recall: {13:.7f} Vali F1 score: {14:.7f} Test Loss: {15:.7f} Test Acc: {16:.7f} Test AUPRC: {17:.7f} Test AUC: {18:.7f} Test Prcision: {19:.7f} Test Recall: {20:.7f} Test F1 score: {21:.7f}".format(epoch + 1, train_loss, train_accuracy, train_auprc, train_auc, train_precision, train_recall, train_F1, vali_loss, vali_accuracy, vali_auprc, vali_auc, vali_precision, vali_recall, vali_F1, test_loss, test_accuracy, test_auprc, test_auc, test_precision, test_recall, test_F1))

            if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
                early_stopping(-vali_auprc, model, path)
            elif args.data == 'PAM' or args.data == 'activity':
                early_stopping(-vali_accuracy, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    accelerator.wait_for_everyone()
    # if accelerator.is_local_main_process:
    #    path = './checkpoints'  # unique checkpoint saving path
    #    del_files(path)  # delete checkpoint files
    #    accelerator.print('success delete checkpoints')

    if accelerator.is_local_main_process:
        # plot and save metrics
        directory = f'./results/{setting}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Plotting accuracy
        plt.figure(figsize=(10, 4))
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(vali_accuracies, label='Validation Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()
        plt.savefig(f'./results/{setting}/accuracy_plot.png')
        plt.close()

        # Plotting loss
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(vali_losses, label='Validation Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.savefig(f'./results/{setting}/loss_plot.png')
        plt.close()

        # criterion = nn.CrossEntropyLoss()
        # # mae_metric = nn.L1Loss()
        # metric = "accuracy"
        # test(args, accelerator, model, test_loader, dim, setting)
else:
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_nc{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des,
        args.num_classes, ii)

    # train_data, train_loader = data_provider(args, 'train')
    # test_data, test_loader = data_provider(args, 'test')
    #
    # if args.model == 'Autoformer':
    #     model = Autoformer.Model(args).float()
    # elif args.model == 'DLinear':
    #     model = DLinear.Model(args).float()
    # else:
    #     model = TimeLLM.Model(args).float()
    #
    # train_steps = len(train_loader)  # The number of data points in each batch
    # early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
    #
    # trained_parameters = []
    # for p in model.parameters():
    #     if p.requires_grad is True:
    #         trained_parameters.append(p)
    #
    # model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    #
    # if args.lradj == 'COS':
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    # else:
    #     scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
    #                                         steps_per_epoch=train_steps,
    #                                         pct_start=args.pct_start,
    #                                         epochs=args.train_epochs,
    #                                         max_lr=args.learning_rate)
    #
    # criterion = nn.CrossEntropyLoss()
    # # mae_metric = nn.L1Loss()
    # metric = "accuracy"
    #
    # test_loader, model, model_optim, scheduler = accelerator.prepare(test_loader, model, model_optim, scheduler)
    # args.content = load_content(args)
    # criterion = nn.CrossEntropyLoss()
    # metric = "accuracy"
    #
    # print("loading model")
    # print(setting)
    # model = accelerator.unwrap_model(model)
    # model.load_state_dict(
    #     torch.load(os.path.join('./checkpoints/' + setting + '-' + args.model_comment, 'checkpoint'), map_location=torch.device('cuda:0')))
    # # accelerator.load_state(os.path.join('./checkpoints/' + setting + '-' + args.model_comment, 'checkpoint'))
    # test(args, accelerator, model, test_loader, criterion, metric, setting, test=1)

    args.content = load_content(args)
    print(args.content)

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    elif args.model == 'TimeLLM_4':
        model = TimeLLM_4.Model(args).float()
    elif args.model == 'TimeLLM_5':
        model = TimeLLM_5.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    test(args, accelerator, model, test_loader, dim, setting)
