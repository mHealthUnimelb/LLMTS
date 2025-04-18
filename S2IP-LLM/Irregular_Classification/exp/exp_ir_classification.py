from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, adjust_model
from utils.metrics import metric
import torch
import torch.nn as nn
from models import S2IPLLM
from torch.nn.utils import clip_grad_norm_
from utils.losses import mape_loss, mase_loss, smape_loss

from transformers import AdamW

from torch.utils.data import Dataset, DataLoader
from torch import optim
import os
import time
import warnings
import numpy as np
from sklearn.metrics import average_precision_score, auc, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm

warnings.filterwarnings('ignore')


class Exp_ir_Classification(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'S2IPLLM': S2IPLLM,

        }

        self.device = torch.device('cuda:0')
        self.model = self._build_model()

        self.all_data, _ = data_provider(args, None)
        self.train_data = self.all_data.data_objects["train_data"]
        self.val_data = self.all_data.data_objects["val_data"]
        self.test_data = self.all_data.data_objects["test_data"]
        self.train_loader = self.all_data.data_objects["train_dataloader"]
        self.vali_loader = self.all_data.data_objects["val_dataloader"]
        self.test_loader = self.all_data.data_objects["test_dataloader"]
        self.dim = self.all_data.data_objects["input_dim"]
        # self.train_data, self.train_loader = self._get_data(flag='train')
        # self.vali_data, self.vali_loader = self._get_data(flag='val')
        # self.test_data, self.test_loader = self._get_data(flag='test')
        # self.dim = self.train_data.data_objects["input_dim"]

        self.optimizer = self._select_optimizer()
        self.train_criterion = self._select_criterion(flag='train')
        self.vali_criterion = self._select_criterion(flag='vali')
        self.test_criterion = self._select_criterion(flag='test')

        self.train_accuracies = []
        self.vali_accuracies = []
        self.test_accuracies = []
        self.train_losses = []
        self.vali_losses = []
        self.test_losses = []
        self.train_auprcs = []
        self.vali_auprcs = []
        self.test_auprcs = []

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
        return model

    def _get_data(self, flag):
        data_set, _ = data_provider(self.args, flag)
        return data_set, None

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _calculate_class_weights(self, y_true):
        # print(f"y_true label: {np.unique(y_true)}, type: {type(y_true)}")
        # print(f"num_classes label: {np.arange(num_classes)}, type: {type(np.arange(num_classes))}")
        # y_true_copy = copy.deepcopy(y_true)
        # y_true_copy = np.array(y_true_copy)
        y_true_copy = np.array(y_true)
        print(f"y_true_copy label: {np.unique(y_true_copy)}, type: {type(y_true_copy)}")
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_true_copy), y=y_true_copy)
        return torch.tensor(class_weights, dtype=torch.float).to(self.device)

    def _select_criterion(self, flag):
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss == 'SMAPE':
            criterion = smape_loss()
        elif self.args.loss == 'CE':
            if flag == 'train':
                y_train = [label for (_, _, _, _, label) in self.train_data]
                class_weights = self._calculate_class_weights(y_train)
                print("class_weights", class_weights)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            elif flag == 'vali':
                y_val = [label for (_, _, _, _, label) in self.val_data]
                class_weights = self._calculate_class_weights(y_val)
                print("class_weights", class_weights)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            elif flag == 'test':
                y_test = [label for (_, _, _, _, label) in self.test_data]
                class_weights = self._calculate_class_weights(y_test)
                print("class_weights", class_weights)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()

        return criterion

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
            train_preds = []
            train_trues = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in tqdm(enumerate(self.train_loader)):
                iter_count += 1
                self.optimizer.zero_grad()
                batch_x = batch_x[:, :, :self.dim]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().long().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x)[0]
                        else:
                            outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        loss = self.train_criterion(outputs, batch_y)

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x)[0]
                    else:
                        outputs, res = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    loss = self.train_criterion(outputs, batch_y)

                train_loss.append(loss.item())
                simlarity_losses.append(res['simlarity_loss'].item())

                loss += self.args.sim_coef * res['simlarity_loss']

                train_preds.append(outputs)
                train_trues.append(batch_y)

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
            self.train_losses.append(train_loss)
            sim_loss = np.average(simlarity_losses)
            # vali_loss = self.vali(self.vali_data, self.vali_loader, self.criterion)

            train_preds = torch.cat(train_preds, 0)
            train_trues = torch.cat(train_trues, 0)
            train_probs = torch.nn.functional.softmax(train_preds)
            train_predictions = torch.argmax(train_probs, dim=1)
            train_trues = train_trues.flatten()
            # calculate AUPRC
            # train_auprc = self._select_metric(train_probs, train_trues)
            # calculate accuracy
            correct = (train_predictions == train_trues).float()
            train_trues = train_trues.detach().cpu().numpy()
            train_auc = roc_auc_score(train_trues,
                                      train_probs.detach().cpu().numpy()[:, 1]) if not self.args.classify_pertp else 0.
            train_auprc = average_precision_score(train_trues, train_probs.detach().cpu().numpy()[:,
                                                               1]) if not self.args.classify_pertp else 0.
            train_accuracy = correct.mean().item()
            self.train_auprcs.append(train_auprc)
            self.train_accuracies.append(train_accuracy)

            vali_loss, vali_accuracy, vali_auprc, vali_auc = self.vali(self.vali_loader,
                                                                       self.vali_criterion)
            self.vali_losses.append(vali_loss)
            self.vali_accuracies.append(vali_accuracy)
            self.vali_auprcs.append(vali_auprc)
            test_loss, test_accuracy, test_auprc, test_auc = self.vali(self.test_loader,
                                                                       self.train_criterion)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)
            self.test_auprcs.append(test_auprc)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Sim Loss: {4:.7f} \
            Train Accuracy: {5:.7f} Train AUPRC: {6:.7f} Train AUC: {7:.7f} Vali Accuracy: {8:.7f} Vali AUPRC: {9:.7f} \
            Vali AUC: {10:.7f} Test Accuracy: {11:.7f} Test AUPRC: {12:.7f} Test AUC: {13:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, sim_loss, train_accuracy, train_auprc, train_auc,
                vali_accuracy, vali_auprc, vali_auc, test_accuracy, test_auprc, test_auc))

            prev_best = early_stopping.best_score
            early_stopping(-vali_auc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if early_stopping.best_score != prev_best:
                best_ckpt = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
                self.model.load_state_dict(torch.load(best_ckpt))
                test_loss, test_accuracy, test_auprc, test_auc = self.vali(self.test_loader,
                                                                           self.vali_criterion)
                print("New best valid -> test set results at epoch {}: acc {:.4f}, auprc {:.4f}, auc {:.4f}"
                      .format(epoch + 1, test_accuracy, test_auprc, test_auc))

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            adjust_model(self.model, epoch + 1, self.args)

    def vali(self, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x[:, :, :self.dim]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().long().to(self.device)

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
                        outputs, res = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.detach().cpu().item())

                preds.append(outputs.detach().cpu())
                trues.append(batch_y.detach().cpu())

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten()
        trues = trues.cpu().numpy()
        accuracy = np.mean(predictions == trues)
        auc = roc_auc_score(trues, probs.cpu().numpy()[:, 1]) if not self.args.classify_pertp else 0.
        auprc = average_precision_score(trues, probs.cpu().numpy()[:, 1]) if not self.args.classify_pertp else 0.

        self.model.train()
        return total_loss, accuracy, auprc, auc

    def test(self, setting, test=0):
        if test:
            print("Loading model")
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            print(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))

        preds = []
        trues = []

        # sim_matrix = []
        # input_embedding = []
        # prompted_embedding = []
        # last_embedding = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm(enumerate(self.test_loader)):
                batch_x = batch_x[:, :, :self.dim]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().long().to(self.device)

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
                        outputs, res = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.float().detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten()
        # auprc = self._select_metric(probs, trues)
        trues = trues.cpu().numpy()
        accuracy = np.mean(predictions == trues)
        auc = roc_auc_score(trues, probs.cpu().numpy()[:, 1]) if not self.args.classify_pertp else 0.
        auprc = average_precision_score(trues, probs.cpu().numpy()[:, 1]) if not self.args.classify_pertp else 0.

        # print('Accuracy:{}'.format(accuracy))
        # print('AUPRC:{}'.format(auprc))
        # print('AUC:{}'.format(auc))
        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        f = open(os.path.join(folder_path, "result_classification.txt"), 'a')
        f.write(setting + "  \n")
        f.write('Accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('AUPRC:{}'.format(auprc))
        f.write('\n')
        f.write('AUC:{}'.format(auc))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return accuracy, auprc, auc

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    #
    #     preds = []
    #     trues = []
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
    #             batch_x = batch_x[:, :, :self.dim]
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().long().to(self.device)
    #
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x)[0]
    #                     else:
    #                         outputs = self.model(batch_x)
    #             else:
    #                 if self.args.output_attention:
    #                     outputs = self.model(batch_x)[0]
    #                 else:
    #                     outputs, res = self.model(batch_x)
    #             f_dim = -1 if self.args.features == 'MS' else 0
    #
    #             pred = outputs
    #             true = batch_y
    #
    #             preds.append(outputs)
    #             trues.append(true)
    #
    #     preds = torch.cat(preds, 0)
    #     trues = torch.cat(trues, 0)
    #
    #     probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
    #     predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    #     trues = trues.flatten()
    #     trues = trues.cpu().numpy()
    #     accuracy = np.mean(predictions == trues)
    #     auc = roc_auc_score(trues, probs.cpu().numpy()[:, 1]) if not self.args.classify_pertp else 0.
    #     auprc = average_precision_score(trues, probs.cpu().numpy()[:, 1]) if not self.args.classify_pertp else 0.
    #
    #     print('Accuracy:{}'.format(accuracy))
    #     print('AUPRC:{}'.format(auprc))
    #     print('AUC:{}'.format(auc))
    #     # result save
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #     f = open(os.path.join(folder_path, "result_classification.txt"), 'a')
    #     f.write(setting + "  \n")
    #     f.write('Accuracy:{}'.format(accuracy))
    #     f.write('\n')
    #     f.write('AUPRC:{}'.format(auprc))
    #     f.write('\n')
    #     f.write('AUC:{}'.format(auc))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()
    #
    #     # Saving true labels and predictions
    #     # np.savetxt(os.path.join(folder_path, 'test_trues.txt'), trues, fmt='%d')
    #     # np.savetxt(os.path.join(folder_path, 'test_predictions.txt'), predictions, fmt='%d')
    #     return