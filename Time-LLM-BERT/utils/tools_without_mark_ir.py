import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from torchmetrics.classification import MulticlassAveragePrecision
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, average_precision_score, ConfusionMatrixDisplay, precision_recall_curve, \
    auc, roc_auc_score, precision_score, recall_score
from data_provider.data_factory import data_provider
from accelerate.state import DistributedType

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


# def auprc_metric(num_class, probs, target):
#     probs = probs.float().detach().cpu().numpy()
#     target = target.float().detach().cpu().numpy()
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


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, input_dim, vali_loader, criterion, metric):
    dim = input_dim
    total_loss = []
    all_logits = []
    trues = []
    preds = []
    # auprc_metric = MulticlassAveragePrecision(num_classes=args.num_classes, average="macro")
    # total_mae_loss = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in tqdm(enumerate(vali_loader)):
            observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:, :, -1]

            batch_x = observed_data.float().to(accelerator.device)
            batch_y = batch_y.squeeze().long().to(accelerator.device)

            print("batch data memory allocated in vali:", torch.cuda.memory_allocated() / (1024 ** 3))
            print("batch data memory reserved in vali:", torch.cuda.memory_reserved() / (1024 ** 3))
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]["aligned_logits"]
                        outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                        # outputs = model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                    else:
                        # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)["aligned_logits"]
                        outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
                        # outputs = model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)
            else:
                if args.output_attention:
                    # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]["aligned_logits"]
                    outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                    # outputs = model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                else:
                    # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)["aligned_logits"]
                    outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
                    # outputs = model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)
                # outputs = outputs.to(torch.float32)
            print("model output memory allocated in vali:", torch.cuda.memory_allocated() / (1024 ** 3))
            print("model output memory reserved in vali:", torch.cuda.memory_reserved() / (1024 ** 3))

            print(f"outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
            print(f"batch_y shape: {batch_y.shape}, dtype: {batch_y.dtype}")            

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
            print("gather outputs and y memory allocated in vali:", torch.cuda.memory_allocated() / (1024 ** 3))
            print("gather outputs and y memory reserved in vali:", torch.cuda.memory_reserved() / (1024 ** 3))
            # outputs = accelerator.gather_for_metrics(outputs)
            # batch_y = accelerator.gather_for_metrics(batch_y)

            # f_dim = -1 if args.features == 'MS' else 0
            # outputs = outputs[:, -args.pred_len:, f_dim:]
            # batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            print(f"outputs shape: {outputs.shape}, type: {outputs.dtype}")

            if args.classify_pertp:
                outputs = outputs.reshape(-1, args.num_classes)
                batch_y = batch_y.argmax(-1).reshape(-1)

            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())

            # prob = torch.nn.functional.softmax(outputs.detach())
            # pred = torch.argmax(prob, dim=1)
            # true = batch_y.detach()

            all_logits.append(outputs.detach())
            trues.append(batch_y.detach())

            # mae_loss = mae_metric(pred, true)
    total_loss = np.average(total_loss)

    all_logits = torch.cat(all_logits, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(all_logits)
    # auprc = auprc_metric(args.num_classes, probs, trues)
    # auprc = auprc_metric(probs, trues)
    predictions = torch.argmax(probs, dim=1).cpu().numpy()
    trues = trues.flatten().cpu().numpy()

    if metric == "accuracy":
        accuracy = cal_accuracy(predictions, trues)

    if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
        auc = roc_auc_score(trues, probs.cpu().float().numpy()[:, 1]) if not args.classify_pertp else 0.
        auprc = average_precision_score(trues, probs.cpu().float().numpy()[:, 1]) if not args.classify_pertp else 0.
    elif args.data == 'PAM' or args.data == 'activity':
        auc = roc_auc_score(one_hot(trues), probs.detach().cpu().float().numpy())
        auprc = average_precision_score(one_hot(trues),
                                                probs.detach().cpu().float().numpy())
        precision = precision_score(trues, probs.detach().cpu().float().numpy().argmax(1),
                                            average='macro', )
        recall = recall_score(trues, probs.detach().cpu().float().numpy().argmax(1),
                                    average='macro', )
        F1 = 2 * (precision * recall) / (
                precision + recall)
    # total_mae_loss = np.average(total_mae_loss)

    model.train()
    # return total_loss, total_mae_loss
    if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
        return total_loss, accuracy, auc, auprc
    elif args.data == 'PAM' or args.data == 'activity':
        return total_loss, accuracy, auc, auprc, precision, recall, F1

# def test(args, accelerator, model, test_loader, criterion, metric, setting, test=0):
#     # test_data, test_loader = data_provider(args, 'test')
#     # test_loader, model = accelerator.prepare(test_loader, model)
#     # if test:
#     #     print("loading model")
#     #     print(setting)
#     #     model = accelerator.unwrap_model(model)
#     #     model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + '-' + args.model_comment, 'checkpoint')))
#     #     # accelerator.load_state(os.path.join('./checkpoints/' + setting + '-' + args.model_comment, 'checkpoint'))
#
#     total_loss = []
#     all_logits = []
#     trues = []
#     preds = []
#     auprc_metric = MulticlassAveragePrecision(num_classes=args.num_classes, average="macro")
#     # total_mae_loss = []
#
#     model.eval()
#     with torch.no_grad():
#         for i, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
#             batch_x = batch_x.float().to(accelerator.device)
#             batch_y = batch_y.squeeze().long().to(accelerator.device)
#
#             # encoder - decoder
#             if args.use_amp:
#                 with torch.cuda.amp.autocast():
#                     if args.output_attention:
#                         outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
#                     else:
#                         outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
#             else:
#                 if args.output_attention:
#                     outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
#                 else:
#                     outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
#                 # outputs = outputs.to(torch.float32)
#
#             print(f"outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
#             print(f"batch_y shape: {batch_y.shape}, dtype: {batch_y.dtype}")
#             outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
#             # outputs = accelerator.gather_for_metrics(outputs)
#             # batch_y = accelerator.gather_for_metrics(batch_y)
#
#             # f_dim = -1 if args.features == 'MS' else 0
#             # outputs = outputs[:, -args.pred_len:, f_dim:]
#             # batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
#
#             print(f"outputs shape: {outputs.shape}, type: {outputs.dtype}")
#             # loss = criterion(outputs, batch_y)
#             # total_loss.append(loss.item())
#
#             # prob = torch.nn.functional.softmax(outputs.detach())
#             # pred = torch.argmax(prob, dim=1)
#             # true = batch_y.detach()
#
#             all_logits.append(outputs.detach())
#             trues.append(batch_y.detach())
#
#             # mae_loss = mae_metric(pred, true)
#     # total_loss = np.average(total_loss)
#
#     all_logits = torch.cat(all_logits, 0)
#     trues = torch.cat(trues, 0)
#     probs = torch.nn.functional.softmax(all_logits)
#     auprc = auprc_metric(probs, trues)
#     predictions = torch.argmax(probs, dim=1).cpu().numpy()
#     trues = trues.flatten().cpu().numpy()
#
#     if metric == "accuracy":
#         accuracy = cal_accuracy(predictions, trues)
#
#
#     # total_mae_loss = np.average(total_mae_loss)
#
#     # return total_loss, total_mae_loss
#     accelerator.print(
#         "Test Acc: {0:.7f} Test AUPRC: {1:.7f}".format(accuracy, auprc))
#     return

# def test(args, model, test_loader, input_dim, setting, test=0):
#     # test_data, test_loader = data_provider(args, 'test')
#     # test_loader, model = accelerator.prepare(test_loader, model)
#     # if test:
#     #     print("loading model")
#     #     print(setting)
#     #     model = accelerator.unwrap_model(model)
#     #     model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + '-' + args.model_comment, 'checkpoint')))
#     #     # accelerator.load_state(os.path.join('./checkpoints/' + setting + '-' + args.model_comment, 'checkpoint'))
#
#     dim = input_dim
#     print("loading model")
#     print(setting)
#     model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + '-' + args.model_comment, 'checkpoint'), map_location=torch.device('cuda:0')))
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model = model.to(torch.bfloat16)
#
#     total_loss = []
#     all_logits = []
#     trues = []
#     preds = []
#     # auprc_metric = MulticlassAveragePrecision(num_classes=args.num_classes, average="macro")
#     # total_mae_loss = []
#
#     model.eval()
#     with torch.no_grad():
#         for i, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
#             batch_x = batch_x[:, :, :dim].float().to(device)
#             batch_y = batch_y.squeeze().long().to(device)
#
#             # encoder - decoder
#             if args.use_amp:
#                 with torch.cuda.amp.autocast():
#                     if args.output_attention:
#                         # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]["aligned_logits"]
#                         outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
#                     else:
#                         # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)["aligned_logits"]
#                         outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
#             else:
#                 if args.output_attention:
#                     # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]["aligned_logits"]
#                     outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
#                 else:
#                     # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)["aligned_logits"]
#                     outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
#                 # outputs = outputs.to(torch.float32)
#
#             print(f"outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
#             print(f"batch_y shape: {batch_y.shape}, dtype: {batch_y.dtype}")
#             # outputs = accelerator.gather_for_metrics(outputs)
#             # batch_y = accelerator.gather_for_metrics(batch_y)
#
#             # f_dim = -1 if args.features == 'MS' else 0
#             # outputs = outputs[:, -args.pred_len:, f_dim:]
#             # batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
#
#             print(f"outputs shape: {outputs.shape}, type: {outputs.dtype}")
#             # loss = criterion(outputs, batch_y)
#             # total_loss.append(loss.item())
#
#             # prob = torch.nn.functional.softmax(outputs.detach())
#             # pred = torch.argmax(prob, dim=1)
#             # true = batch_y.detach()
#
#             all_logits.append(outputs.detach())
#             trues.append(batch_y.detach())
#
#             # mae_loss = mae_metric(pred, true)
#     # total_loss = np.average(total_loss)
#
#     all_logits = torch.cat(all_logits, 0)
#     trues = torch.cat(trues, 0)
#     probs = torch.nn.functional.softmax(all_logits)
#     # auprc = auprc_metric(args.num_classes, probs, trues)
#     # auprc = auprc_metric(probs, trues)
#     predictions = torch.argmax(probs, dim=1).cpu().numpy()
#     trues = trues.flatten().cpu().numpy()
#     auc = roc_auc_score(trues, probs.cpu().float().numpy()[:, 1]) if not args.classify_pertp else 0.
#     auprc = average_precision_score(trues, probs.cpu().float().numpy()[:, 1]) if not args.classify_pertp else 0.
#     accuracy = cal_accuracy(predictions, trues)
#
#
#     # total_mae_loss = np.average(total_mae_loss)
#
#     # return total_loss, total_mae_loss
#     print("Test Acc: {0:.7f} Test AUROC: {1:.7f} Test AUPRC: {2:.7f}".format(accuracy, auc, auprc))
#
#     # resuqlt save
#     folder_path = './results/' + setting + '/'
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#
#     # Compute Confusion Matrix
#     cm = confusion_matrix(trues, predictions)
#     # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(self.args.num_class)))
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot()
#     plt.title(f'Confusion Matrix')
#     plt.savefig(f'./results/{setting}/confusion_matrix.png')
#     plt.close()
#
#     return


def test(args, accelerator, model, test_loader, input_dim, setting):
    # test_data, test_loader = data_provider(args, 'test')
    # test_loader, model = accelerator.prepare(test_loader, model)
    # if test:
    #     print("loading model")
    #     print(setting)
    #     model = accelerator.unwrap_model(model)
    #     model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + '-' + args.model_comment, 'checkpoint')))
    #     # accelerator.load_state(os.path.join('./checkpoints/' + setting + '-' + args.model_comment, 'checkpoint'))

    dim = input_dim
    print("loading model")
    # print(setting)
    best_model_path = './checkpoints/' + setting + '-' + args.model_comment + '/checkpoint'
    print("best_model_path", best_model_path)
    # if accelerator.state.distributed_type != DistributedType.NO:
    #     accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    unwrapped_model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage))
    unwrapped_model = unwrapped_model.bfloat16()
    unwrapped_model.to(accelerator.device)

    total_loss = []
    all_logits = []
    trues = []
    preds = []
    # auprc_metric = MulticlassAveragePrecision(num_classes=args.num_classes, average="macro")
    # total_mae_loss = []

    unwrapped_model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
            observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:, :, -1]

            batch_x = observed_data.float().to(accelerator.device)
            batch_y = batch_y.squeeze().long().to(accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]["aligned_logits"]
                        # outputs = unwrapped_model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                        outputs = unwrapped_model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                    else:
                        # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)["aligned_logits"]
                        # outputs = unwrapped_model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)
                        outputs = unwrapped_model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
            else:
                if args.output_attention:
                    # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]["aligned_logits"]
                    # outputs = unwrapped_model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                    outputs = unwrapped_model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)[0]
                else:
                    # outputs = model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)["aligned_logits"]
                    # outputs = unwrapped_model(batch_x, observed_mask, x_mark_enc=None, x_dec=None, x_mark_dec=None)
                    outputs = unwrapped_model(batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None)
                # outputs = outputs.to(torch.float32)

            print(f"outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
            print(f"batch_y shape: {batch_y.shape}, dtype: {batch_y.dtype}")
            # outputs = accelerator.gather_for_metrics(outputs)
            # batch_y = accelerator.gather_for_metrics(batch_y)

            # f_dim = -1 if args.features == 'MS' else 0
            # outputs = outputs[:, -args.pred_len:, f_dim:]
            # batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            if args.classify_pertp:
                outputs = outputs.reshape(-1, args.num_classes)
                batch_y = batch_y.argmax(-1).reshape(-1)

            print(f"outputs shape: {outputs.shape}, type: {outputs.dtype}")
            # loss = criterion(outputs, batch_y)
            # total_loss.append(loss.item())

            # prob = torch.nn.functional.softmax(outputs.detach())
            # pred = torch.argmax(prob, dim=1)
            # true = batch_y.detach()

            all_logits.append(outputs.detach())
            trues.append(batch_y.detach())

            # mae_loss = mae_metric(pred, true)
    # total_loss = np.average(total_loss)

    all_logits = torch.cat(all_logits, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(all_logits)
    # auprc = auprc_metric(args.num_classes, probs, trues)
    # auprc = auprc_metric(probs, trues)
    predictions = torch.argmax(probs, dim=1).cpu().numpy()
    trues = trues.flatten().cpu().numpy()
    accuracy = cal_accuracy(predictions, trues)

    if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
        auc = roc_auc_score(trues, probs.cpu().float().numpy()[:, 1]) if not args.classify_pertp else 0.
        auprc = average_precision_score(trues, probs.cpu().float().numpy()[:, 1]) if not args.classify_pertp else 0.
    elif args.data == 'PAM' or args.data == 'activity':
        auc = roc_auc_score(one_hot(trues), probs.detach().cpu().float().numpy())
        auprc = average_precision_score(one_hot(trues),
                                                probs.detach().cpu().float().numpy())
        precision = precision_score(trues, probs.detach().cpu().float().numpy().argmax(1),
                                            average='macro', )
        recall = recall_score(trues, probs.detach().cpu().float().numpy().argmax(1),
                                    average='macro', )
        F1 = 2 * (precision * recall) / (
                precision + recall)

    # total_mae_loss = np.average(total_mae_loss)
    if accelerator.is_local_main_process:
        # return total_loss, total_mae_loss
        if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
            print("Test Acc: {0:.7f} Test AUROC: {1:.7f} Test AUPRC: {2:.7f}".format(accuracy, auc, auprc))
        elif args.data == 'PAM' or args.data == 'activity':
            print("Test Acc: {0:.7f} Test AUROC: {1:.7f} Test AUPRC: {2:.7f} Test Precision: {3:.7f} Test Recall: {4:.7f} Test F1: {5:.7f}".format(accuracy, auc, auprc, precision, recall, F1))

        # resuqlt save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Compute Confusion Matrix
        cm = confusion_matrix(trues, predictions)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(self.args.num_class)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix')
        plt.savefig(f'./results/{setting}/confusion_matrix.png')
        plt.close()

    return


# def test(args, accelerator, model, train_loader, vali_loader, criterion):
#     x, _ = train_loader.dataset.last_insample_window()
#     y = vali_loader.dataset.timeseries
#     x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
#     x = x.unsqueeze(-1)
#
#     model.eval()
#     with torch.no_grad():
#         B, _, C = x.shape
#         dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
#         dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
#         outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
#         id_list = np.arange(0, B, args.eval_batch_size)
#         id_list = np.append(id_list, B)
#         for i in range(len(id_list) - 1):
#             outputs[id_list[i]:id_list[i + 1], :, :] = model(
#                 x[id_list[i]:id_list[i + 1]],
#                 None,
#                 dec_inp[id_list[i]:id_list[i + 1]],
#                 None
#             )
#         accelerator.wait_for_everyone()
#         outputs = accelerator.gather_for_metrics(outputs)
#         f_dim = -1 if args.features == 'MS' else 0
#         outputs = outputs[:, -args.pred_len:, f_dim:]
#         pred = outputs
#         true = torch.from_numpy(np.array(y)).to(accelerator.device)
#         batch_y_mark = torch.ones(true.shape).to(accelerator.device)
#         true = accelerator.gather_for_metrics(true)
#         batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)
#
#         loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)
#
#     model.train()
#     return loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content

def one_hot(y_):
        # Function to encode output labels from number indexes
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        y_ = y_.reshape(len(y_))

        y_ = [int(x) for x in y_]
        n_values = np.max(y_) + 1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]
