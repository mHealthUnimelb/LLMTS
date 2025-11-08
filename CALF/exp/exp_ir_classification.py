from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from utils.cmLoss import cmLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAveragePrecision
from sklearn.metrics import confusion_matrix, average_precision_score, ConfusionMatrixDisplay, precision_recall_curve, \
    auc, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import seaborn as sns
import copy

warnings.filterwarnings('ignore')


class Exp_IR_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_IR_Classification, self).__init__(args)
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
        # model input depends on data
        self.all_data, _ = self._get_data(flag=None)
        self.train_data = self.all_data.data_objects["train_data"]
        self.val_data = self.all_data.data_objects["val_data"]
        self.test_data = self.all_data.data_objects["test_data"]
        self.train_loader = self.all_data.data_objects["train_dataloader"]
        self.vali_loader = self.all_data.data_objects["val_dataloader"]
        self.test_loader = self.all_data.data_objects["test_dataloader"]
        self.args.num_class = len(self.all_data.class_names)
        self.args.dim = self.all_data.data_objects["input_dim"]
        # model init
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        param_dict = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' in n], "lr": 1e-4},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' not in n],
             "lr": self.args.learning_rate}
        ]
        model_optim = optim.Adam([param_dict[1]], lr=self.args.learning_rate)
        loss_optim = optim.Adam([param_dict[0]], lr=self.args.learning_rate)

        return model_optim, loss_optim

    def _calculate_class_weights(self, y_true, num_classes):
        y_true_copy = np.array(y_true)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_true_copy), y=y_true_copy)
        return torch.tensor(class_weights, dtype=torch.float).to(self.device)

    def _select_criterion(self):
        criterion = cmLoss(self.args.feature_loss,
                           self.args.output_loss,
                           self.args.task_loss,
                           self.args.task_name,
                           self.args.feature_w,
                           self.args.output_w,
                           self.args.task_w)
        return criterion

    def _select_vali_criterion(self):
        return nn.CrossEntropyLoss()

    def _one_hot(self, y_):
        # Function to encode output labels from number indexes
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        y_ = y_.reshape(len(y_))

        y_ = [int(x) for x in y_]
        n_values = np.max(y_) + 1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]

    def train(self, setting):
        dim = self.all_data.data_objects["input_dim"]

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, loss_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_preds = []
            train_trues = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label) in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()
                loss_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                batch_len = batch_x.shape[0]
                observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:,
                                                                                                             :, -1]

                outputs = self.model(observed_data)

                if self.args.classify_pertp:
                    outputs["outputs_time"] = outputs["outputs_time"].reshape(-1, self.args.num_class)
                    outputs["outputs_text"] = outputs["outputs_text"].reshape(-1, self.args.num_class)
                    label = label.argmax(-1).reshape(-1)
                    loss = criterion(outputs, label.long())
                else:
                    loss = criterion(outputs, label.long().squeeze(-1))
                
                loss.backward()
                model_optim.step()
                loss_optim.step()

                train_loss.append(loss.item())

                train_preds.append(outputs["outputs_time"])
                train_trues.append(label)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            self.train_losses.append(train_loss)

            train_preds = torch.cat(train_preds, 0)
            train_trues = torch.cat(train_trues, 0)
            train_probs = torch.nn.functional.softmax(train_preds)
            train_predictions = torch.argmax(train_probs, dim=1)
            train_trues = train_trues.flatten()
            # calculate accuracy
            correct = (train_predictions == train_trues).float()
            train_trues = train_trues.detach().cpu().numpy()
            if self.args.data == 'P12' or self.args.data == 'P19' or self.args.data == 'eICU' or self.args.data == 'PhysioNet' or self.args.data == 'MIMIC':
                train_auc = roc_auc_score(train_trues,
                                          train_probs.detach().cpu().numpy()[:,
                                          1]) if not self.args.classify_pertp else 0.
                train_auprc = average_precision_score(train_trues, train_probs.detach().cpu().numpy()[:,
                                                                   1]) if not self.args.classify_pertp else 0.
                train_accuracy = correct.mean().item()
                self.train_auprcs.append(train_auprc)
                self.train_accuracies.append(train_accuracy)

                vali_loss, vali_accuracy, vali_auprc, vali_auc = self.vali(self.args, self.vali_loader,
                                                                           self._select_vali_criterion())
                self.vali_losses.append(vali_loss)
                self.vali_accuracies.append(vali_accuracy)
                self.vali_auprcs.append(vali_auprc)
                test_loss, test_accuracy, test_auprc, test_auc = self.vali(self.args, self.test_loader,
                                                                           self._select_vali_criterion())
                self.test_losses.append(test_loss)
                self.test_accuracies.append(test_accuracy)
                self.test_auprcs.append(test_auprc)

                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc: {3:.3f} Train AUPRC: {4:.3f} Train AUC: {5:.3f} Vali Loss: {6:.3f} Vali Acc: {7:.3f} Vali AUPRC: {8:.3f} Vali AUC: {9:.3f} Test Loss: {10:.3f} Test Acc: {11:.3f} Test AUPRC: {12:.3f} Test AUC: {13:.3f}"
                    .format(epoch + 1, train_steps, train_loss, train_accuracy, train_auprc, train_auc, vali_loss,
                            vali_accuracy, vali_auprc, vali_auc, test_loss, test_accuracy, test_auprc, test_auc))                
            elif self.args.data == 'PAM' or self.args.data == 'activity':
                train_auc = roc_auc_score(self._one_hot(train_trues),
                                          train_probs.detach().cpu().numpy())
                train_auprc = average_precision_score(self._one_hot(train_trues),
                                                      train_probs.detach().cpu().numpy())
                train_precision = precision_score(train_trues, train_probs.detach().cpu().numpy().argmax(1),
                                                  average='macro', )
                train_recall = recall_score(train_trues, train_probs.detach().cpu().numpy().argmax(1),
                                            average='macro', )
                train_F1 = 2 * (train_precision * train_recall) / (
                        train_precision + train_recall)
                train_accuracy = correct.mean().item()

                vali_loss, vali_accuracy, vali_auprc, vali_auc, vali_precision, vali_recall, vali_F1 = self.vali(self.args,
                                                                                           self.vali_loader,
                                                                                           self._select_vali_criterion())

                test_loss, test_accuracy, test_auprc, test_auc, test_precision, test_recall, test_F1 = self.vali(self.args,
                                                                                           self.test_loader,
                                                                                           self._select_vali_criterion())

                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc: {3:.3f} Train AUPRC: {4:.3f} Train AUC: {5:.3f} Train Precision: {6:.3f} Train Recall: {7:.3f} Train F1 score: {8:.3f} Vali Loss: {9:.3f} Vali Acc: {10:.3f} Vali AUPRC: {11:.3f} Vali AUC: {12:.3f} Vali Precision: {13:.3f} Vali Recall: {14:.3f} Vali F1 score: {15:.3f} Test Loss: {16:.3f} Test Acc: {17:.3f} Test AUPRC: {18:.3f} Test AUC: {19:.3f} Test Prcision: {20:.3f} Test Recall: {21:.3f} Test F1 score: {22:.3f}"
                    .format(epoch + 1, train_steps, train_loss, train_accuracy, train_auprc, train_auc, train_precision, train_recall, train_F1,
                            vali_loss, vali_accuracy, vali_auprc, vali_auc, vali_precision, vali_recall, vali_F1, test_loss, test_accuracy,
                            test_auprc, test_auc, test_precision, test_recall, test_F1))

            if self.args.data == 'P12' or self.args.data == 'P19' or self.args.data == 'eICU' or self.args.data == 'MIMIC':
                early_stopping(-vali_auprc, self.model, path)
            elif self.args.data == 'PAM' or self.args.data == 'activity':
                early_stopping(-vali_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # plot and save the training, validation, test accuracy and loss figure
        self.plot_and_save_metrics(setting)

        return self.model

    def vali(self, args, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        dim = self.all_data.data_objects["input_dim"]

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                batch_len = batch_x.shape[0]
                observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:,
                                                                                                             :, -1]

                outputs = self.model(observed_data)
                
                if self.args.classify_pertp:
                    outputs = outputs["outputs_time"].reshape(-1, self.args.num_class)
                    label = label.argmax(-1).reshape(-1)
                    loss = criterion(outputs, label.long())
                else:
                    outputs = outputs["outputs_time"]
                    loss = criterion(outputs, label.long().squeeze(-1))

                total_loss.append(loss.cpu().numpy())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten()
        trues = trues.cpu().numpy()

        if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'PhysioNet' or args.data == 'MIMIC':
            auc = roc_auc_score(trues, probs.cpu().numpy()[:, 1]) if not args.classify_pertp else 0.
            auprc = average_precision_score(trues, probs.cpu().numpy()[:, 1]) if not args.classify_pertp else 0.
        elif args.data == 'PAM' or args.data == 'activity':
            auc = roc_auc_score(self._one_hot(trues), probs.cpu().numpy())
            auprc = average_precision_score(self._one_hot(trues),
                                            probs.cpu().numpy())
            precision = precision_score(trues, probs.cpu().numpy().argmax(1),
                                        average='macro', )
            recall = recall_score(trues, probs.cpu().numpy().argmax(1),
                                  average='macro', )
            F1 = 2 * (precision * recall) / (precision + recall)
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'PhysioNet' or args.data == 'MIMIC':
            return total_loss, accuracy, auprc, auc
        elif args.data == 'PAM' or args.data == 'activity':
            return total_loss, accuracy, auprc, auc, precision, recall, F1

    def test(self, args, setting, test=0):
        dim = self.all_data.data_objects["input_dim"]
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                batch_len = batch_x.shape[0]
                observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:,
                                                                                                             :, -1]

                outputs = self.model(observed_data)
                
                if self.args.classify_pertp:
                    outputs = outputs["outputs_time"].reshape(-1, self.args.num_class)
                    label = label.argmax(-1).reshape(-1)
                else:
                    outputs = outputs["outputs_time"]

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten()
        trues = trues.cpu().numpy()

        if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'PhysioNet' or args.data == 'MIMIC':
            auc = roc_auc_score(trues, probs.cpu().numpy()[:, 1]) if not args.classify_pertp else 0.
            auprc = average_precision_score(trues, probs.cpu().numpy()[:, 1]) if not args.classify_pertp else 0.
            accuracy = cal_accuracy(predictions, trues)
        elif args.data == 'PAM' or args.data == 'activity':
            auc = roc_auc_score(self._one_hot(trues), probs.cpu().numpy())
            auprc = average_precision_score(self._one_hot(trues),
                                            probs.cpu().numpy())
            accuracy = cal_accuracy(predictions, trues)
            precision = precision_score(trues, probs.cpu().numpy().argmax(1),
                                        average='macro', )
            recall = recall_score(trues, probs.cpu().numpy().argmax(1),
                                  average='macro', )
            F1 = 2 * (precision * recall) / (precision + recall)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Compute Confusion Matrix
        cm = confusion_matrix(trues, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        sns.set_context("poster")
        fig, ax = plt.subplots(figsize=(20, 15))
        disp.plot(ax=ax)
        plt.title(f'Confusion Matrix for {setting}')
        plt.savefig(f'./results/{setting}/confusion_matrix.png')
        plt.close()

        if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'MIMIC':
            print('Accuracy:{}'.format(accuracy))
            print('AUPRC:{}'.format(auprc))
            print('AUC:{}'.format(auc))
            
            file_name = 'result_classification.txt'
            f = open(os.path.join(folder_path, file_name), 'a')
            f.write(setting + "  \n")
            f.write('Accuracy:{}'.format(accuracy))
            f.write('\n')
            f.write('AUPRC:{}'.format(auprc))
            f.write('\n')
            f.write('AUROC:{}'.format(auc))
            f.write('\n')
            f.write('\n')
            f.close()
        elif args.data == 'PAM' or args.data == 'activity':
            print('Accuracy:{}'.format(accuracy))
            print('AUPRC:{}'.format(auprc))
            print('AUC:{}'.format(auc))
            print('Precision:{}'.format(precision))
            print('Recall:{}'.format(recall))
            print('F1 score:{}'.format(F1))

            file_name = 'result_classification.txt'
            f = open(os.path.join(folder_path, file_name), 'a')
            f.write(setting + "  \n")
            f.write('Accuracy:{}'.format(accuracy))
            f.write('\n')
            f.write('AUPRC:{}'.format(auprc))
            f.write('\n')
            f.write('AUROC:{}'.format(auc))
            f.write('\n')
            f.write('Precision:{}'.format(precision))
            f.write('\n')
            f.write('Recall:{}'.format(recall))
            f.write('\n')
            f.write('F1 score:{}'.format(F1))
            f.write('\n')
            f.write('\n')
            f.close()

        return

    def plot_and_save_metrics(self, setting):
        directory = f'./results/{setting}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Plotting accuracy
        plt.figure(figsize=(10, 4))
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.vali_accuracies, label='Validation Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()
        plt.savefig(f'./results/{setting}/accuracy_plot.png')
        plt.close()

        # Plotting loss
        plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.vali_losses, label='Validation Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.savefig(f'./results/{setting}/loss_plot.png')
        plt.close()
