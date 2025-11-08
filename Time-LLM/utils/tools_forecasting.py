import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm

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


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_loader):
    total_loss = []
    total_mae_loss = []

    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(vali_loader)):
            seen_data, seen_tp, seen_mask = batch['observed_data'], batch['observed_tp'], batch['observed_mask'],
            future_data, future_tp, future_mask = batch['data_to_predict'], batch['tp_to_predict'], batch['mask_predicted_data']  
            # print('checking data shape', seen_data.shape, future_data.shape)

            ### the original shape is [batch_size, seq_len, feature], pad the se1_dim to 512
            batch_size, seq_len_seen, feature_dim = seen_data.shape
            batch_size, seq_len_future, feature_dim = future_data.shape
            seen_data_padded = torch.zeros(batch_size, 512, feature_dim, device=seen_data.device, dtype=seen_data.dtype)
            future_data_padded = torch.zeros(batch_size, 512, feature_dim, device=future_data.device, dtype=future_data.dtype)

            # Copy the existing data
            seen_data_padded[:, :seq_len_seen, :] = seen_data
            future_data_padded[:, :seq_len_future, :] = future_data

            # Do the same for masks
            seen_mask_padded = torch.zeros(batch_size, 512, feature_dim, device=seen_mask.device, dtype=seen_mask.dtype)
            future_mask_padded = torch.zeros(batch_size, 512, feature_dim, device=future_mask.device, dtype=future_mask.dtype)
            future_mask_padded[:, :seq_len_future, :] = future_mask

            # Use these padded tensors
            batch_x = seen_data_padded.to(accelerator.device).float()
            batch_y = future_data_padded.to(accelerator.device).float()
            
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x)[0]
                    else:
                        outputs = model(batch_x)
            else:
                if args.output_attention:
                    outputs = model(batch_x)[0]
                else:
                    outputs = model(batch_x)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            # pred = outputs.detach()
            # true = batch_y.detach()

            if future_mask.sum() > 0:
                future_mask_padded = future_mask_padded.to(accelerator.device)
                mse_loss = ((batch_y - outputs)**2) * future_mask_padded
                mse_loss = mse_loss.sum() / future_mask_padded.sum()
                # mse_loss = F.mse_loss(outputs['outputs_time'], batch_y, reduction='none')
                loss = mse_loss
                # mae_loss = mae_metric(outputs, batch_y)
                mae_loss = (batch_y - outputs).abs()
                mae_loss = mae_loss * future_mask_padded
                mae_loss = mae_loss.sum() / future_mask_padded.sum()

                # Skip extreme values
                if mse_loss.item() < 1000 and not torch.isnan(mse_loss) and not torch.isinf(mse_loss):
                    total_loss.append(loss.detach().cpu().item())
                    total_mae_loss.append(mae_loss.detach().cpu().item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content