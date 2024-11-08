import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize



def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode, ana_ratio):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _,_ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        # valid_loss, valid_acc = 0, 0 # for pendigits
        # valid_loss, valid_acc, _, _= model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    print('start to evaluate the model after save the fine-tuned model')
    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _,auprc = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        # test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f} | AUPRC      : {auprc:0.4f}')
        # logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

        os.makedirs(os.path.join(experiment_log_dir, f"saved_models_{ana_ratio}"), exist_ok=True)
        chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
        torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models_{ana_ratio}", f'ckp_last.pt'))

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2, adf_label) in enumerate(train_loader):
        
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # print('input data shape:',data.shape) #([128, 2, 2500])

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            # selecte the samples based on the original 
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            # print('feature shape:',features1.shape) #torch.Size([128, 128, 315])
            # print('predictions shape:',predictions1.shape) 

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            # print('feature shape:',features1.shape) 

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1 
            zjs = temp_cont_lstm_feat2 

        else:
            output = model(data)

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2


            # print('loss here:', loss )

            
        else: # supervised training or fine tuining
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    probas = np.array([]) # for storing predicted probabilities

    with torch.no_grad():
        for data, labels, _, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                # get the softmax probabilities
                softmax_probs = torch.nn.functional.softmax(predictions, dim=1).cpu().numpy()
                if probas.size == 0:
                    probas = softmax_probs
                else:
                    probas = np.vstack((probas, softmax_probs))


            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())
                # trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc

        # compute AUPRC
        n_classes = len(np.unique(trgs)) # number of classes
        # print('here', n_classes)
        n_classes = 4
        trgs_bin = label_binarize(trgs, classes=np.arange(n_classes)) # binarize labels
        auprc = average_precision_score(trgs_bin, probas) # compute AUPRC

    return total_loss, total_acc, outs, trgs, auprc
    # return total_loss, total_acc, outs, trgs




def model_evaluate_simclr(model, config, test_dl, device, training_mode):
    model.eval()
    # temporal_contr_model.eval()

    total_loss = []
    total_acc = []
    total_prc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    probas = np.array([]) # for storing predicted probabilities

    with torch.no_grad():
        for data, labels, _, _,_  in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())
                # n_classes = config.num_classes_target
                n_classes = config.num_classes
                onehot_label = torch.zeros(len(labels), n_classes)
                for i in range(len(labels)):
                    onehot_label[i, labels[i]] = 1

                # print('onehot label shape:', onehot_label.shape)

                pred_numpy = predictions.detach().cpu().numpy()
                # print('shape:', onehot_label.shape, pred_numpy.shape)

                prc= average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro")
                total_prc.append(prc)

                '''calculate the auprc for ECG dataset..'''
                # # get the softmax probabilities
                softmax_probs = torch.nn.functional.softmax(predictions, dim=1).cpu().numpy()

                if probas.size == 0:
                    probas = softmax_probs
                else:
                    probas = np.vstack((probas, softmax_probs))

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())
                

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        # total_prc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
        # total_prc = torch.tensor(total_prc).mean()
        trgs_bin = label_binarize(trgs, classes=np.arange(n_classes)) # binarize labels
        auprc = average_precision_score(trgs_bin, probas) # compute AUPRC

    return total_loss, total_acc, outs, trgs,auprc