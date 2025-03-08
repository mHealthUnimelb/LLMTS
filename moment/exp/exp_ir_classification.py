from momentfm import MOMENTPipeline
from data_provider.data_factory import data_provider
from tqdm import tqdm
import os
import torch
import numpy as np
from sklearn.metrics import average_precision_score, auc, roc_auc_score

class Exp_ir_Classification(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0')
        self.model = self._build_model()

        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')
        self.dim = self.train_data.data_objects["input_dim"]

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _build_model(self):
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "classification",
                "n_channels": 41,
                "num_class": 2
            },
        )
        model.init()
        model.to(self.device)
        return model

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')

        probs = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
                batch_x = batch_x[:, :, :self.dim]
                batch_x = batch_x.permute(0, 2, 1).contiguous()
                print("batch_x shape: ", batch_x.shape)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().long().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(x_enc=batch_x)[0]
                        else:
                            outputs = self.model(x_enc=batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(x_enc=batch_x)[0]
                    else:
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

        # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten()
        # auprc = self._select_metric(probs, trues)
        trues = trues.cpu().numpy()
        accuracy = np.mean(predictions == trues)
        auc = roc_auc_score(trues, probs.cpu().numpy()[:, 1]) if not self.args.classify_pertp else 0.
        auprc = average_precision_score(trues, probs.cpu().numpy()[:, 1]) if not self.args.classify_pertp else 0.

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('Accuracy:{}'.format(accuracy))
        print('AUPRC:{}'.format(auprc))
        print('AUC:{}'.format(auc))
        f = open("result_classification.txt", 'a')
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