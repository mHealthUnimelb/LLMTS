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
    auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import seaborn as sns
import copy

warnings.filterwarnings('ignore')

class Hook:
    def __init__(self):
        self.output = None
        self.attention_weights = None

    def hook_fn(self, module, input, output):
        # This function will be called when the layer produces an output
        self.output = output

    def attention_hook_fn(self, module, input, output):
        self.attention_weights = output[1]

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.train_accuracies = []
        self.vali_accuracies = []
        self.test_accuracies = []
        self.train_losses = []
        self.vali_losses = []
        self.test_losses = []
        self.train_auprcs = []
        self.vali_auprcs = []
        self.test_auprcs = []
        self.hook = Hook()
        self.handle = self.model.in_layer.register_forward_hook(self.hook.hook_fn)
        self.cross_attention_handle = self.model.in_layer.cross_attention.register_forward_hook(self.hook.attention_hook_fn)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_dim
        self.args.num_class = len(train_data.class_names)
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
        print(f"y_true label: {np.unique(y_true)}, type: {type(y_true)}")
        print(f"num_classes label: {np.arange(num_classes)}, type: {type(np.arange(num_classes))}")
        y_true_copy = copy.deepcopy(y_true)
        y_true_copy = y_true_copy.numpy()
        print(f"y_true_copy label: {np.unique(y_true_copy)}, type: {type(y_true_copy)}")
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_true_copy), y=y_true_copy)
        return torch.tensor(class_weights, dtype=torch.float).to(self.device)

    def _select_criterion(self):
        # # extract labels from the training data to compute class weights
        # train_data, _ = self._get_data(flag='train')
        # y_train = train_data.y_data
        #
        # # compute class weights
        # class_weights = self._calculate_class_weights(y_train, self.args.num_class)
        # class_weights = self.train_class_weights

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

    # def _select_metric(self, probs, target):
    #     metric = MulticlassAveragePrecision(num_classes=self.args.num_class, average="macro")
    #     return metric(probs, target)

    def _select_metric(self, probs, target):
        probs = probs.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        # Initialize list to store AUPRC for each class
        auprcs = []

        # Compute AUPRC for each class
        for i in range(self.args.num_class):
            # For class `i`, the true labels are `1` if the actual label is `i`, else `0`
            precision, recall, _ = precision_recall_curve(target == i, probs[:, i])
            auprc = auc(recall, precision)
            auprcs.append(auprc)

        # Return the average AUPRC across all classes
        return np.mean(auprcs)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, loss_optim = self._select_optimizer()
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

            # Store initial weights before applying LoRA
            initial_weights = {name: param.clone() for name, param in self.model.named_parameters() if
                               param.requires_grad}

            for i, (batch_x, label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                loss_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                # print("batch_x shape: ", batch_x.shape)
                # print("label shape: ", label.shape)

                outputs = self.model(batch_x)

                loss = criterion(outputs, label.long().squeeze(-1))
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

                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
                loss_optim.step()

                # monitored_layer = dict(self.model.named_parameters())[monitored_layer_name]

            # Compare weights after applying LoRA
            for name, param in self.model.named_parameters():
                if name in initial_weights:
                    if not torch.equal(param, initial_weights[name]):
                        print(f"Weights changed in layer: {name}")
                    else:
                        print(f"No change in layer: {name}")

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
            train_auprc = self._select_metric(train_probs, train_trues)
            # calculate accuracy
            correct = (train_predictions == train_trues).float()
            train_accuracy = correct.mean().item()
            self.train_auprcs.append(train_auprc)
            self.train_accuracies.append(train_accuracy)

            vali_loss, vali_accuracy, vali_auprc = self.vali(vali_data, vali_loader, self._select_vali_criterion())
            self.vali_losses.append(vali_loss)
            self.vali_accuracies.append(vali_accuracy)
            self.vali_auprcs.append(vali_auprc)
            test_loss, test_accuracy, test_auprc = self.vali(test_data, test_loader, self._select_vali_criterion())
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)
            self.test_auprcs.append(test_auprc)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc: {3:.3f} Train AUPRC: {4:.3f} Vali Loss: {5:.3f} Vali Acc: {6:.3f} Vali AUPRC: {7:.3f} Test Loss: {8:.3f} Test Acc: {9:.3f} Test AUPRC: {10:.3f}"
                .format(epoch + 1, train_steps, train_loss, train_accuracy, train_auprc, vali_loss, vali_accuracy,
                        vali_auprc, test_loss, test_accuracy, test_auprc))

            # if self.args.cos:
            #     scheduler.step()
            #     print("lr = {}".format(model_optim.param_groups[0]['lr']))
            # else:
            #     adjust_learning_rate(model_optim, epoch + 1, self.args)

            early_stopping(-vali_accuracy, self.model, path)
            # early_stopping(vali_loss, self.model, path)
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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)["outputs_time"]

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze(-1))
                total_loss.append(loss.cpu().numpy())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        # print(f'{vali_data.x_data.shape} shape: {preds.shape} {trues.shape}')
        # print('test shape:', preds.shape, trues.shape)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten()
        auprc = self._select_metric(probs, trues)
        trues = trues.cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # Saving true labels and predictions
        np.savetxt('./results/vali_trues.txt', trues, fmt='%d')
        np.savetxt('./results/vali_predictions.txt', predictions, fmt='%d')

        self.model.train()
        return total_loss, accuracy, auprc

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            # self.model.load_state_dict(torch.load('./checkpoints/classification_ECG_CALF_2500__CALF_ECG_ftM_sl2500_ll0_pl0_dm768_nh4_el2_dl1_df768_fc1_ebtimeF_dtTrue_test_gpt6_0/checkpoint.pth'))

        preds = []
        trues = []
        time_embeddings = []
        text_embeddings = []
        cross_attention_weights = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)["outputs_time"]
                outputs_time1, outputs_text1 = self.hook.output  #store the embedding
                cross_attention_weights.append(self.hook.attention_weights.cpu())
                print("attention shape: ", self.hook.attention_weights.cpu().shape)

                time_embeddings.append(outputs_time1.cpu())
                text_embeddings.append(outputs_text1.cpu())

                preds.append(outputs.detach())
                trues.append(label)

                # for name, param in self.model.in_layer.named_parameters():
                #     print(f"Layer: {name} | Size: {param.size()}")
                #
                # print(f"word embedding shape: {self.model.in_layer.word_embedding.shape} value: {self.model.in_layer.word_embedding}")

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        self.handle.remove()  # remove the hook after use
        self.cross_attention_handle.remove()
        cross_attention_weights = torch.cat(cross_attention_weights, 0)
        print("attention shape: ", cross_attention_weights.shape)
        time_embeddings = torch.cat(time_embeddings, 0)
        time_channedl_1_embedding = time_embeddings[:, 0, :]
        time_channedl_2_embedding = time_embeddings[:, 1, :]
        text_embeddings = torch.cat(text_embeddings, 0)
        text_channel_1_embedding = text_embeddings[:, 0, :]
        text_channel_2_embedding = text_embeddings[:, 1, :]
        print("time embedding shape: ", time_channedl_1_embedding.shape)
        print("text embedding shape: ", text_channel_1_embedding.shape)

        # print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten()
        auprc = self._select_metric(probs, trues)
        trues = trues.cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.visualize_embeddings(time_channedl_1_embedding, trues, title="time_channel_1_token_embedding",
                                  setting=setting)
        self.visualize_embeddings(time_channedl_2_embedding, trues, title="time_channel_2_token_embedding",
                                  setting=setting)

        self.visualize_embeddings(text_channel_1_embedding, trues, title="aligned_text_channel_1_token_embedding",
                                  setting=setting)
        self.visualize_embeddings(text_channel_2_embedding, trues, title="aligned_text_channel_2_token_embedding",
                                  setting=setting)

        words = ["Trend", "seasonality", "cyclicity", "rise", "peak", "pattern", "shift", "position", "irregular",
                 "missing", "inconsistent", "discontinuous", "heart", "period", "echo", "arm", "key", "mint"]

        self.plot_attention_weights(cross_attention_weights, words_list=words, title="dropped_cross_attention_map", setting=setting)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Compute Confusion Matrix
        cm = confusion_matrix(trues, predictions)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(self.args.num_class)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix for {setting}')
        plt.savefig(f'./results/{setting}/confusion_matrix.png')
        plt.close()

        print('accuracy:{}'.format(accuracy))
        print('AUPRC:{}'.format(auprc))
        file_name = 'result_classification.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('AUPRC:{}'.format(auprc))
        f.write('\n')
        f.write('\n')
        f.close()

        # Saving true labels and predictions
        np.savetxt(os.path.join(folder_path, 'test_trues.txt'), trues, fmt='%d')
        np.savetxt(os.path.join(folder_path, 'test_predictions.txt'), predictions, fmt='%d')
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

    def visualize_embeddings(self, embeddings, labels, title='t-SNE plot of Word Embeddings', setting=None):
        """
        Visualize word embeddings using t-SNE.

        Args:
        - embeddings (Tensor): Word embeddings to visualize.
        - labels (list or Tensor): Corresponding labels for the embeddings.
        - title (str): Title for the plot.
        """
        # convert embeddings to numpy array
        embeddings_np = embeddings.cpu().numpy()

        # apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings_np)

        # create a scatter plot
        directory = f'./results/{setting}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        sns.set_context("poster")
        plt.figure(figsize=(12, 10))
        # Get unique labels and plot each with a different color/marker for legend
        # label_mapping = {0: 'AFIB', 1: 'AFL', 2: 'J', 3: 'N'}
        # labels_mapped = [label_mapping[label] for label in labels]
        unique_labels = np.unique(labels)
        # colors = ['r', 'g', 'b', 'c']
        # cmap = ListedColormap(colors[:len(np.unique(labels))])

        # colors = plt.cm.get_cmap('viridis', len(unique_labels))  # Use colormap to generate colors for each label
        #
        # for i, label in enumerate(unique_labels):
        #     indices = labels == label
        #     # Plot points for each category on the same plot with different colors
        #     plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
        #                 color=colors(i), label=f'Class {label}', alpha=0.7)

        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)

        # add legend with unique labels
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)),
                              markersize=10) for label in unique_labels]
        # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8, label=f'Class {label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles, unique_labels, title="Labels", loc='upper left', markerscale=2.2, fontsize=45)
        # plt.legend(title='Class', handles=handles, loc='best')

        # cbar = plt.colorbar(scatter)
        # cbar.set_label('Class')
        # plt.title(title)
        # Remove ticks
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.xlabel('Dimension 1', fontsize=45)
        plt.ylabel('Dimension 2', fontsize=45)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # plt.legend(title='Categories', loc='best')
        plt.tight_layout()
        plt.savefig(f'./results/{setting}/{title}.png')
        plt.close()

    # def plot_attention_weights(self, attention_weights, words_list, title='Cross Attention Map', setting=None):
    #     """Plot the cross-attention weights captured by the hook."""
    #     print("attention_weights shape: ", attention_weights.shape)
    #     # attention_weights_np = attention_weights.mean(dim=1).numpy()  # Average across channel
    #     attention_weights_np = attention_weights[:, 1, :].numpy()  # Select the first channel
    #
    #     sns.set_context("poster", font_scale=1.2)
    #     plt.figure(figsize=(22, 28))
    #     # sns.heatmap(attention_weights_np, cmap='viridis', cbar=True)
    #     plt.imshow(attention_weights_np, aspect='auto', cmap='viridis', interpolation='nearest')
    #     plt.xticks(ticks=np.arange(len(words_list)), labels=words_list, rotation=90, fontsize=42)
    #     plt.yticks(ticks=np.arange(10), labels=list(range(1, 11)))
    #
    #     cbar = plt.colorbar()
    #     cbar.set_label('Relevance Score', rotation=270, labelpad=42)
    #     plt.xlabel('Selected Words')
    #     plt.ylabel("Time Series Instances")
    #     # plt.title(title)
    #     plt.savefig(f'./results/{setting}/{title}.png')
    #     plt.close()

    # rot90
    # def plot_attention_weights(self, attention_weights, words_list, title='Cross Attention Map', setting=None):
    #     """Plot the cross-attention weights captured by the hook."""
    #     print("attention_weights shape: ", attention_weights.shape)
    #     # Select the first channel
    #     attention_weights_np = attention_weights[:, 1, :].numpy()
    #
    #     # Rotate the attention weights 90 degrees clockwise
    #     attention_weights_np = np.rot90(attention_weights_np, k=-1)  # or k=3
    #
    #     sns.set_context("poster", font_scale=1)
    #     plt.figure(figsize=(30, 10))  # Adjusted figsize for the rotated plot
    #
    #     # Plot the rotated attention weights
    #     plt.imshow(attention_weights_np, aspect='auto', cmap='viridis', interpolation='nearest')
    #     plt.xticks(ticks=np.arange(10), labels=list(range(1, 11)))  # Updated to match the new orientation
    #     plt.yticks(ticks=np.arange(len(words_list)), labels=words_list, rotation=0, fontsize=28)  # Adjusted rotation
    #
    #     cbar = plt.colorbar()
    #     cbar.set_label('Relevance Score', rotation=270, labelpad=28)
    #     plt.xlabel('Time Series Instances')
    #     # plt.ylabel("Selected Words")
    #     # plt.title(title)
    #     plt.savefig(f'./results/{setting}/{title}.png')
    #     plt.close()

    # without rot90
    def plot_attention_weights(self, attention_weights, words_list, title='Cross Attention Map', setting=None):
        """Plot the cross-attention weights captured by the hook."""
        print("attention_weights shape: ", attention_weights.shape)
        # Select the first channel
        attention_weights_np = attention_weights[:, 1, :].numpy()

        # # Rotate the attention weights 90 degrees clockwise
        # attention_weights_np = np.rot90(attention_weights_np, k=-1)  # or k=3

        sns.set_context("poster", font_scale=1.4)
        plt.figure(figsize=(35, 80))  # Adjusted figsize for the rotated plot

        # Plot the rotated attention weights
        plt.imshow(attention_weights_np, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.xticks(ticks=np.arange(len(words_list)), labels=words_list, rotation=90, fontsize=80)
        plt.yticks(ticks=np.arange(10), labels=list(range(1, 11)), fontsize=80)

        cbar = plt.colorbar()
        cbar.set_label('Relevance Score', rotation=270, labelpad=120, fontsize=80)
        cbar.ax.tick_params(labelsize=65)
        # plt.xlabel("Selected Words")
        plt.ylabel('Time Series Instances', fontsize=80)
        # plt.title(title)
        plt.savefig(f'./results/{setting}/{title}.png')
        plt.close()
