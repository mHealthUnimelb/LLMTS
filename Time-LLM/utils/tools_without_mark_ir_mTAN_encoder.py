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
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.squeeze().long().to(accelerator.device)
            observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:, :, -1]

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(torch.cat((observed_data, observed_mask), 2), observed_tp, x_mark_enc=None, x_dec=None, x_mark_dec=None, device=accelerator.device)[0]
                    else:
                        outputs = model(torch.cat((observed_data, observed_mask), 2), observed_tp, x_mark_enc=None, x_dec=None, x_mark_dec=None, device=accelerator.device)
            else:
                if args.output_attention:
                    outputs = model(torch.cat((observed_data, observed_mask), 2), observed_tp, x_mark_enc=None, x_dec=None, x_mark_dec=None, device=accelerator.device)[0]
                else:
                    outputs = model(torch.cat((observed_data, observed_mask), 2), observed_tp, x_mark_enc=None, x_dec=None, x_mark_dec=None, device=accelerator.device)        

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
        
            if args.classify_pertp:
                outputs = outputs.reshape(-1, args.num_classes)
                batch_y = batch_y.argmax(-1).reshape(-1)

            loss = criterion(outputs, batch_y)
            total_loss.append(loss.item())

            all_logits.append(outputs.detach())
            trues.append(batch_y.detach())

    total_loss = np.average(total_loss)

    all_logits = torch.cat(all_logits, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(all_logits)
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

    model.train()
    
    if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
        return total_loss, accuracy, auc, auprc
    elif args.data == 'PAM' or args.data == 'activity':
        return total_loss, accuracy, auc, auprc, precision, recall, F1


def test(args, accelerator, model, test_loader, input_dim, setting):
    dim = input_dim
    print("loading model")
    best_model_path = './checkpoints/' + setting + '-' + args.model_comment + '/checkpoint'
    
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
    
    unwrapped_model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.squeeze().long().to(accelerator.device)
            observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:, :, -1]

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = unwrapped_model(torch.cat((observed_data, observed_mask), 2), observed_tp, x_mark_enc=None, x_dec=None, x_mark_dec=None, device=accelerator.device)[0]
                    else:
                        outputs = unwrapped_model(torch.cat((observed_data, observed_mask), 2), observed_tp, x_mark_enc=None, x_dec=None, x_mark_dec=None, device=accelerator.device)
            else:
                if args.output_attention:
                    outputs = unwrapped_model(torch.cat((observed_data, observed_mask), 2), observed_tp, x_mark_enc=None, x_dec=None, x_mark_dec=None, device=accelerator.device)[0]
                else:
                    outputs = unwrapped_model(torch.cat((observed_data, observed_mask), 2), observed_tp, x_mark_enc=None, x_dec=None, x_mark_dec=None, device=accelerator.device)

            if args.classify_pertp:
                outputs = outputs.reshape(-1, args.num_classes)
                batch_y = batch_y.argmax(-1).reshape(-1)

            all_logits.append(outputs.detach())
            trues.append(batch_y.detach())

    all_logits = torch.cat(all_logits, 0)
    trues = torch.cat(trues, 0)
    probs = torch.nn.functional.softmax(all_logits)
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

    if accelerator.is_local_main_process:
        if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
            print("Test Acc: {0:.7f} Test AUROC: {1:.7f} Test AUPRC: {2:.7f}".format(accuracy, auc, auprc))
        elif args.data == 'PAM' or args.data == 'activity':
            print("Test Acc: {0:.7f} Test AUROC: {1:.7f} Test AUPRC: {2:.7f} Test Precision: {3:.7f} Test Recall: {4:.7f} Test F1: {5:.7f}".format(accuracy, auc, auprc, precision, recall, F1))

        # result save
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
