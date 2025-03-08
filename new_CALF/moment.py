from momentfm import MOMENTPipeline
import torch
import torch.nn as nn
import argparse
import numpy as np
import random
from data_provider import utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--dataset', type=str, default='physionet')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--quantization', type=float, default=0.1,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
args = parser.parse_args()

# def test(test_data, test_loader, model):
#     preds = []
#     trues = []
#
#     model.eval()
#     with torch.no_grad():
#         for i, (batch_x, label) in enumerate(test_loader):



if __name__ == '__main__':
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)
    elif args.dataset == 'mimiciii':
        data_obj = utils.get_mimiciii_data(args)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]

    # model
    model = MOMENTPipeline.from_pretrained(
        "AutoLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'classification',
            'n_channels': 41,
            'num_class': 2
        },
    )
    model.init()

    preds = []
    trues = []
    for test_batch, label in test_loader:
        output = model(x_enc = test_batch[:, :, :dim])
        preds.append(output)
        trues.append(label)
    # preds = torch.cat(preds, 0)
    # trues = torch.cat(trues, 0)
    print(preds)


    # optimizer = torch.optim.Adam(model.params, lr=args.lr)
    # criterion = nn.CrossEntropyLoss()
    #
    # for train_batch, label in train_loader:
    #     train_batch, label = train_batch.to(device), label.to(device)  # torch.Size([128, 190, 83]) torch.Size([128])
    #     # observed_data, observed_mask, observed_tp \
    #     #     = train_batch[:, :, :dim], train_batch[:, :, dim:2 * dim], train_batch[:, :,
    #     #                                                                -1]  # observed_data: (128, 190, 41) observed_mask: (128, 190, 41) observed_tp: (128, 190)
    #
    #     # output = model(x_enc=(torch.cat((observed_data, observed_mask), 2), observed_tp))
    #     output = model(x_enc=train_batch)
    #
    #     loss = criterion(output.logits, label)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     print(f"loss: {loss.item():.3f}")
