from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,adjust_model
from utils.metrics import metric
import torch
import torch.nn as nn
from models import  S2IPLLM
from torch.nn.utils import clip_grad_norm_
from utils.losses import mape_loss, mase_loss, smape_loss

from transformers import AdamW

from torch.utils.data import Dataset, DataLoader
from torch import optim
import os
import time
import warnings
import numpy as np

from tqdm import tqdm

from .config import Config
from .data_loader_full import load_forecasting_data

warnings.filterwarnings('ignore')


class Exp_IR_Forecast(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'S2IPLLM': S2IPLLM,
            
        }

        self.device = torch.device('cuda:0')
        self.model = self._build_model()
        
        # self.train_data, self.train_loader = self._get_data(flag='train')
        # self.vali_data, self.vali_loader = self._get_data(flag='val')
        # self.test_data, self.test_loader = self._get_data(flag='test')

        # self.all_data = data_provider(args, None)
        # self.train_loader = self.all_data["train_dataloader"]
        # self.vali_loader = self.all_data["val_dataloader"]
        # self.test_loader = self.all_data["test_dataloader"]
        
        self.train_loader, self.vali_loader, self.test_loader = self._get_data(flag=None)

        self.optimizer = self._select_optimizer()
        # self.criterion = self._select_criterion()

      

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
        

        return model

    def _get_data(self, flag):
        # data_set, data_loader = data_provider(self.args, flag)

        config = Config(self.args)

        data_obj = load_forecasting_data(config)
        train_loader = data_obj["train_dataloader"]
        val_loader = data_obj["val_dataloader"]
        test_loader = data_obj["test_dataloader"]

        return train_loader, val_loader, test_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # def _select_criterion(self):
    #     if self.args.loss=='MSE':
    #         criterion = nn.MSELoss()
    
    #     elif self.args.loss=='SMAPE':
    #         criterion = smape_loss()

    #     return criterion
    

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            simlarity_losses = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in tqdm(enumerate(self.train_loader)):
                iter_count += 1
                self.optimizer.zero_grad()
                # batch_x = batch_x.float().to(self.device)

                # batch_y = batch_y.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                seen_data, seen_tp, seen_mask = batch['observed_data'], batch['observed_tp'], batch['observed_mask'],
                future_data, future_tp, future_mask = batch['data_to_predict'], batch['tp_to_predict'], batch['mask_predicted_data']

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
                batch_x = seen_data_padded.to(self.device).float()
                batch_y = future_data_padded.to(self.device).float()

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x)[0]
                        else:
                            outputs = self.model(batch_x)

                        # f_dim = -1 if self.args.features == 'MS' else 0
                        # outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable]
                        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                        # loss = self.criterion(outputs, batch_y)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x)[0]
                    else:
                        outputs,res = self.model(batch_x)
                        
                f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                # loss = self.criterion(outputs, batch_y)

                future_mask_padded = future_mask_padded.to(self.device)
                mse_loss = ((batch_y - outputs)**2) * future_mask_padded
                mse_loss = ((batch_y - outputs)**2)
                mse_loss = mse_loss.sum() / future_mask_padded.sum()
                loss = mse_loss
                
                train_loss.append(loss.item())
                simlarity_losses.append(res['simlarity_loss'].item())

                loss += self.args.sim_coef*res['simlarity_loss']
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    loss.backward()
                    self.optimizer.step()
                else:
                 
                    loss.backward()
                    self.optimizer.step()
               
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            sim_loss = np.average(simlarity_losses)
            vali_loss, vali_mae_loss = self.vali(self.vali_loader)
            test_loss, test_mae_loss = self.vali(self.test_loader)
    
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali MAE Loss: {4:.7f} Sim Loss: {5:.7f} Test Loss: {6:.7f} Test MAE Loss: {7:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_mae_loss, sim_loss, test_loss, test_mae_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            adjust_model(self.model, epoch + 1,self.args)


    def vali(self, vali_loader):
        total_loss = []
        total_mae_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(vali_loader)):
                # batch_x = batch_x.float().to(self.device)
                # batch_y = batch_y.float().to(self.device)
            
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                seen_data, seen_tp, seen_mask = batch['observed_data'], batch['observed_tp'], batch['observed_mask'],
                future_data, future_tp, future_mask = batch['data_to_predict'], batch['tp_to_predict'], batch['mask_predicted_data']  
                
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
                batch_x = seen_data_padded.to(self.device).float()
                batch_y = future_data_padded.to(self.device).float()

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(torch.bfloat16).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x)[0]
                        else:
                            outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x)[0]
                    else:
                        outputs,res = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0

                # outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)

                # outputs = outputs.detach().cpu()
                # batch_y = batch_y.detach().cpu()

                # loss = criterion(pred, true)
                if future_mask.sum() > 0:
                    future_mask_padded = future_mask_padded.to(self.device)
                    mse_loss = ((batch_y - outputs)**2) * future_mask_padded
                    mse_loss = mse_loss.sum() / future_mask_padded.sum()
                    loss = mse_loss

                    mae_loss = (batch_y - outputs).abs()
                    mae_loss = mae_loss * future_mask_padded
                    mae_loss = mae_loss.sum() / future_mask_padded.sum()

                    # Skip extreme values
                    if mse_loss.item() < 1000 and not torch.isnan(mse_loss) and not torch.isinf(mse_loss):
                        total_loss.append(loss.detach().cpu().item())
                        total_mae_loss.append(mae_loss.detach().cpu().item())
        
        total_loss = np.average(total_loss)
        total_mae_loss = np.average(total_mae_loss)
        self.model.train()
        return total_loss, total_mae_loss
            

    def test(self, setting, test=1):        
        preds = []
        trues = []
        total_mse = 0.0
        total_mae = 0.0
        batch_count = 0

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        sim_matrix = []
        input_embedding = []
        prompted_embedding = []
        last_embedding = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader)):
                # batch_x = batch_x.float().to(self.device)
                # batch_y = batch_y.float().to(self.device)

                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
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
                batch_x = seen_data_padded.to(self.device).float()
                batch_y = future_data_padded.to(self.device).float()
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x)[0]
                        else:
                            outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs =  self.model(batch_x)[0]
                    else:
                        outputs,res =  self.model(batch_x)
                        
                f_dim = -1 if self.args.features == 'MS' else 0

                # outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable].float()
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float()
                
                if future_mask.sum() > 0:
                    future_mask_padded = future_mask_padded.to(self.device)
                    mse_loss = ((batch_y - outputs)**2) * future_mask_padded
                    mse_loss = mse_loss.sum() / future_mask_padded.sum()

                    mae_loss = (batch_y - outputs).abs()
                    mae_loss = mae_loss * future_mask_padded
                    mae_loss = mae_loss.sum() / future_mask_padded.sum()

                    # Skip extreme values
                    if mse_loss.item() < 1000 and not torch.isnan(mse_loss) and not torch.isinf(mse_loss):
                        total_mse += mse_loss.item()
                        total_mae += mae_loss.item()
                        batch_count += 1

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.float().detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
            
        preds = np.array(preds)
        trues = np.array(trues)
            
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        fina_mse = total_mse / batch_count
        fina_mae = total_mae / batch_count
        print('mse:{}, mae:{}'.format(fina_mse, fina_mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(fina_mse, fina_mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([fina_mae, fina_mse]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return fina_mse, fina_mae
    
