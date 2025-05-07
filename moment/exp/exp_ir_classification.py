from momentfm import MOMENTPipeline
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from tqdm import tqdm
import os
import time
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score, ConfusionMatrixDisplay, precision_recall_curve, \
    auc, roc_auc_score, precision_score, recall_score, f1_score
from utils.tools import EarlyStopping, adjust_learning_rate, adjust_model

class Exp_IR_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_IR_Classification, self).__init__(args)
        self.args = args
        self.all_data, _ = self._get_data(flag=None)
        self.train_loader = self.all_data.data_objects["train_dataloader"]
        self.vali_loader = self.all_data.data_objects["val_dataloader"]
        self.test_loader = self.all_data.data_objects["test_dataloader"]
        self.dim = self.all_data.data_objects["input_dim"]
        self.num_class = self.all_data.num_class
        print("self.num_class", self.num_class)
        self.model = self._build_model()
        self.train_accuracies = []
        self.vali_accuracies = []
        self.test_accuracies = []
        self.train_losses = []
        self.vali_losses = []
        self.test_losses = []
        self.train_auprcs = []
        self.vali_auprcs = []
        self.test_auprcs = []

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _build_model(self):
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small",
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.args.num_variables,
                "num_class": self.args.num_classes,
                "freeze_encoder": True,
                "freeze_embedder": True,
            },
        )
        model.init()
        model.to(self.device)
        print("device:", self.device)
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'CE':
            criterion = nn.CrossEntropyLoss()

        return criterion

    def train(self, setting):
        # train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')
        # all_data = self._get_data(flag=None)
        # train_loader = all_data.data_objects["train_dataloader"]
        # vali_loader = all_data.data_objects["val_dataloader"]
        # test_loader = all_data.data_objects["test_dataloader"]
        dim = self.all_data.data_objects["input_dim"]

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

        # monitored_layer_name = "gpt2.h.0.attn.c_attn.weight"

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_preds = []
            train_trues = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label) in enumerate(self.train_loader):
                num_zeros = (label == 0).sum().item()
                num_ones = (label == 1).sum().item()

                print(f"Batch {i}: # of labels == 0: {num_zeros}, # of labels == 1: {num_ones}")

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:,
                                                                                                             :, -1]
                # time_mask_bool = observed_mask.bool().all(dim=-1)
                # time_mask_int = time_mask_bool.int()
                batch_x = observed_data.permute(0, 2, 1).contiguous()
                print("batch_x shape: ", batch_x.shape)
                label = label.to(self.device)

                print("batch_x shape: ", batch_x.shape)  # 128, 190, 83
                print("label shape: ", label.shape)  # 128

                # outputs = self.model(x_enc=batch_x, input_mask=time_mask_int)
                outputs = self.model(x_enc=batch_x)

                # print(
                #     f"Label min: {label.min()}, Label max: {label.max()}, Label dtype: {label.dtype}, Label shape: {label.shape}")
                # print(f"outputs.shape: {outputs['outputs_time'].shape}, outputs.dtype: {outputs['outputs_time'].dtype}")
                # print(f"label.shape: {label.shape}, label.dtype: {label.dtype}")
                # print(f"label unique values: {torch.unique(label)}")
                loss = criterion(outputs.logits, label.long().squeeze(-1))
                train_loss.append(loss.item())

                train_preds.append(outputs.logits)
                train_trues.append(label)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

                # monitored_layer = dict(self.model.named_parameters())[monitored_layer_name]

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # print(f"Epoch {epoch + 1}, After update: {monitored_layer.data}")

            train_loss = np.average(train_loss)
            self.train_losses.append(train_loss)

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
                                                                           criterion)
                self.vali_losses.append(vali_loss)
                self.vali_accuracies.append(vali_accuracy)
                self.vali_auprcs.append(vali_auprc)
                test_loss, test_accuracy, test_auprc, test_auc = self.vali(self.args, self.test_loader,
                                                                           criterion)
                self.test_losses.append(test_loss)
                self.test_accuracies.append(test_accuracy)
                self.test_auprcs.append(test_auprc)

                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc: {3:.3f} Train AUPRC: {4:.3f} Train AUC: {5:.3f} Vali Loss: {6:.3f} Vali Acc: {7:.3f} Vali AUPRC: {8:.3f} Vali AUC: {9:.3f} Test Loss: {10:.3f} Test Acc: {11:.3f} Test AUPRC: {12:.3f} Test AUC: {13:.3f}"
                    .format(epoch + 1, train_steps, train_loss, train_accuracy, train_auprc, train_auc, vali_loss,
                            vali_accuracy, vali_auprc, vali_auc, test_loss, test_accuracy, test_auprc, test_auc))
            elif self.args.data == 'PAM':
                train_auc = roc_auc_score(self._one_hot(train_trues),
                                          train_probs.detach().cpu().numpy()) if not self.args.classify_pertp else 0.
                train_auprc = average_precision_score(self._one_hot(train_trues),
                                                      train_probs.detach().cpu().numpy()) if not self.args.classify_pertp else 0.
                train_precision = precision_score(train_trues, train_probs.detach().cpu().numpy().argmax(1),
                                                  average='macro', ) if not self.args.classify_pertp else 0.
                train_recall = recall_score(train_trues, train_probs.detach().cpu().numpy().argmax(1),
                                            average='macro', ) if not self.args.classify_pertp else 0.
                train_F1 = 2 * (train_precision * train_recall) / (
                        train_precision + train_recall) if not self.args.classify_pertp else 0.
                train_accuracy = correct.mean().item()

                vali_loss, vali_accuracy, vali_precision, vali_recall, vali_F1 = self.vali(self.args,
                                                                                           self.vali_loader,
                                                                                           self._select_vali_criterion())

                test_loss, test_accuracy, test_precision, test_recall, test_F1 = self.vali(self.args,
                                                                                           self.test_loader,
                                                                                           self._select_vali_criterion())

                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc: {3:.3f} Train Precision: {4:.3f} Train Recall: {5:.3f} Train F1 score: {6:.3f} Vali Loss: {7:.3f} Vali Acc: {8:.3f} Vali Precision: {9:.3f} Vali Recall: {10:.3f} Vali F1 score: {11:.3f} Test Loss: {12:.3f} Test Acc: {13:.3f} Test Prcision: {14:.3f} Test Recall: {15:.3f} Test F1 score: {16:.3f}"
                    .format(epoch + 1, train_steps, train_loss, train_accuracy, train_precision, train_recall, train_F1,
                            vali_loss, vali_accuracy, vali_precision, vali_recall, vali_F1, test_loss, test_accuracy,
                            test_precision, test_recall, test_F1))

            # if self.args.cos:
            #     scheduler.step()
            #     print("lr = {}".format(model_optim.param_groups[0]['lr']))
            # else:
            #     adjust_learning_rate(model_optim, epoch + 1, self.args)

            # early_stopping(-vali_accuracy, self.model, path)
            if self.args.data == 'P12' or self.args.data == 'P19' or self.args.data == 'eICU' or self.args.data == 'MIMIC':
                early_stopping(-vali_auprc, self.model, path)
            elif self.args.data == 'PAM':
                early_stopping(-vali_F1, self.model, path)
            # early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

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
                observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:,
                                                                                                             :, -1]
                # time_mask_bool = observed_mask.bool().all(dim=-1)
                # time_mask_int = time_mask_bool.int()
                batch_x = observed_data.permute(0, 2, 1).contiguous()
                print("batch_x shape: ", batch_x.shape)
                label = label.to(self.device)

                # outputs = self.model(x_enc=batch_x, input_mask=time_mask_int)
                outputs = self.model(x_enc=batch_x)

                pred = outputs.logits.detach()
                loss = criterion(pred, label.long().squeeze(-1))
                total_loss.append(loss.cpu().numpy())

                preds.append(pred)
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        # print(f'{vali_data.x_data.shape} shape: {preds.shape} {trues.shape}')
        # print('test shape:', preds.shape, trues.shape)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten()
        # auprc = self._select_metric(probs, trues)
        trues = trues.cpu().numpy()

        if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'PhysioNet' or args.data == 'MIMIC':
            auc = roc_auc_score(trues, probs.cpu().numpy()[:, 1]) if not args.classify_pertp else 0.
            auprc = average_precision_score(trues, probs.cpu().numpy()[:, 1]) if not args.classify_pertp else 0.
        elif args.data == 'PAM':
            auc = roc_auc_score(self._one_hot(trues), probs.cpu().numpy()) if not args.classify_pertp else 0.
            auprc = average_precision_score(self._one_hot(trues),
                                            probs.cpu().numpy()) if not args.classify_pertp else 0.
            precision = precision_score(trues, probs.cpu().numpy().argmax(1),
                                        average='macro', ) if not args.classify_pertp else 0.
            recall = recall_score(trues, probs.cpu().numpy().argmax(1),
                                  average='macro', ) if not args.classify_pertp else 0.
            F1 = 2 * (precision * recall) / (precision + recall) if not args.classify_pertp else 0.
        accuracy = np.mean(predictions == trues)

        # Saving true labels and predictions
        # np.savetxt('./results/vali_trues.txt', trues, fmt='%d')
        # np.savetxt('./results/vali_predictions.txt', predictions, fmt='%d')

        self.model.train()
        if args.data == 'P12' or args.data == 'P19' or args.data == 'eICU' or args.data == 'PhysioNet' or args.data == 'MIMIC':
            return total_loss, accuracy, auprc, auc
        elif args.data == 'PAM':
            return total_loss, accuracy, precision, recall, F1


    def test(self, setting, test=0):
        dim = self.all_data.data_objects["input_dim"]

        if test:
            print("Loading model")
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            print(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))

        probs = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm(enumerate(self.test_loader)):
                batch_x = batch_x.float().to(self.device)
                observed_data, observed_mask, observed_tp = batch_x[:, :, :dim], batch_x[:, :, dim:2 * dim], batch_x[:,
                                                                                                             :, -1]
                # time_mask_bool = observed_mask.bool().all(dim=-1)
                # time_mask_int = time_mask_bool.int()
                batch_x = observed_data.permute(0, 2, 1).contiguous()
                print("batch_x shape: ", batch_x.shape)
                batch_y = batch_y.float().long().to(self.device)

                # outputs = self.model(x_enc=batch_x, input_mask=time_mask_int)
                outputs = self.model(x_enc=batch_x)

                # pred = outputs
                prob = outputs.logits
                print("prob shape", prob.shape)
                true = batch_y

                # preds.append(pred)
                probs.append(prob)
                trues.append(true)

        # preds = torch.cat(preds, 0)
        probs = torch.cat(probs, 0)
        trues = torch.cat(trues, 0)

        probs = torch.nn.functional.softmax(probs)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten()
        # auprc = self._select_metric(probs, trues)
        trues = trues.cpu().numpy()
        accuracy = np.mean(predictions == trues)

        if self.args.data == 'P12' or self.args.data == 'P19' or self.args.data == 'eICU' or self.args.data == 'PhysioNet' or self.args.data == 'MIMIC':
            auc = roc_auc_score(trues, probs.cpu().numpy()[:, 1]) if not self.args.classify_pertp else 0.
            auprc = average_precision_score(trues, probs.cpu().numpy()[:, 1]) if not self.args.classify_pertp else 0.
        elif self.args.data == 'PAM':
            auc = roc_auc_score(self._one_hot(trues), probs.cpu().numpy()) if not self.args.classify_pertp else 0.
            auprc = average_precision_score(self._one_hot(trues),
                                            probs.cpu().numpy()) if not self.args.classify_pertp else 0.
            precision = precision_score(trues, probs.cpu().numpy().argmax(1),
                                        average='macro', ) if not self.args.classify_pertp else 0.
            recall = recall_score(trues, probs.cpu().numpy().argmax(1),
                                  average='macro', ) if not self.args.classify_pertp else 0.
            F1 = 2 * (precision * recall) / (precision + recall) if not self.args.classify_pertp else 0.

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # print('Accuracy:{}'.format(accuracy))
        # print('AUPRC:{}'.format(auprc))
        # print('AUC:{}'.format(auc))
        if self.args.data == 'P12' or self.args.data == 'P19' or self.args.data == 'eICU' or self.args.data == 'MIMIC':
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
        elif self.args.data == 'PAM':
            f = open(os.path.join(folder_path, "result_classification.txt"), 'a')
            f.write(setting + "  \n")
            f.write('Accuracy:{}'.format(accuracy))
            f.write('\n')
            f.write('Precision:{}'.format(precision))
            f.write('\n')
            f.write('Recall:{}'.format(recall))
            f.write('\n')
            f.write('F1 score:{}'.format(F1))
            f.write('\n')
            f.write('\n')
            f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        if self.args.data == 'P12' or self.args.data == 'P19' or self.args.data == 'eICU' or self.args.data == 'MIMIC':
            return accuracy, auprc, auc
        elif self.args.data == 'PAM':
            return accuracy, precision, recall, F1