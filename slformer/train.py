import torch
import time
import os
from sklearn import metrics
import numpy as np
import pickle as pkl
import csv
import pandas as pd

from util import create_csv

import warnings
warnings.filterwarnings("ignore")



def training_info(loss_record, step):

    info = ""
    for name, loss in loss_record.items():
        info += name
        info += "\t"
        info += str(np.round(loss/step, 4))
        info += "\t"

    return info


def train(device, model, criterion, m, args, train_loader, model_save_path, log_path, test_loader=None, save_model=False, save_log=True, model_class="geneformer"):

    epoch_start_time = time.time()

    best_metric = 0
    best_epoch = 1
    not_improved_count = 0
    best_metric_record = {}, {}

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)

    model = model.to(device)

    for epoch in range(1, args.epochs + 1):

        print('Epoch {}'.format(epoch))

        loss_train_record = {
            "train_loss":0,
            "train_auc": 0,
            "train_aupr": 0,
            "train_f1":0,
            "train_precision":0,
            "train_recall":0,
            "train_acc":0,
        }

        loss_test_record = {
            "test_loss":0,
            "test_auc": 0,
            "test_aupr": 0,
            "test_f1":0,
            "test_precision":0,
            "test_recall":0,
            "test_acc":0,
        }

        average_method = "binary"

        model.train()

        for i, data in enumerate(train_loader):

            if model_class == "geneformer":
                total_emb, label, _, _, _ = data
                total_emb_cuda = torch.autograd.Variable(total_emb.to(device)).to(torch.float32)
            elif model_class == "transformer":
                # sent1, sent2, label, _, _ = data
                sent1, mask1, sent2, mask2, label, _, _, _ = data
                sent1_cuda = sent1.to(device)
                sent2_cuda = sent2.to(device)
                mask1_cuda = mask1.to(device)
                mask2_cuda = mask2.to(device)
            label = label.to(torch.float32)


            optimizer.zero_grad()

                
            if model_class == "geneformer":
                output = model(total_emb_cuda)
                out = torch.squeeze(m(output))
            elif model_class == "transformer":
                # g1, g2, score, att
                # _, _, score, _ = model(sent1_cuda, sent2_cuda)
                output = model(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda)
                # out = torch.squeeze(m(score))
                out = torch.squeeze(m(output))
            
            int_out = np.around(out.detach().cpu().numpy(),0).astype(int)

            
            loss = criterion(out, label.to(device))
            loss = torch.sum(loss)
        
            loss.backward()

            optimizer.step()

            out = out.detach().cpu()

            with torch.no_grad():
                loss_train_record["train_loss"] += loss.item()
                loss_train_record["train_f1"] += metrics.f1_score(label.to(torch.int), int_out, average=average_method)
                loss_train_record["train_precision"] += metrics.precision_score(label.to(torch.int), int_out, average=average_method)
                loss_train_record["train_recall"] += metrics.recall_score(label.to(torch.int), int_out, average=average_method)
                loss_train_record["train_acc"] += metrics.accuracy_score(label.to(torch.int), int_out)
                loss_train_record["train_auc"] += metrics.roc_auc_score(label, out)
                loss_train_record["train_aupr"] += metrics.average_precision_score(label, out)
    

        if test_loader is not None:  # evaluation == False means the model will be trained without evaluation

            model.eval()

            for i, data in enumerate(test_loader):
            
                if model_class == "geneformer":
                    total_emb, label, _, _, _ = data
                    total_emb_cuda = torch.autograd.Variable(total_emb.to(device)).to(torch.float32)
                elif model_class == "transformer":
                    # sent1, sent2, label, _, _ = data
                    sent1, mask1, sent2, mask2, label, _, _, _ = data
                    sent1_cuda = sent1.to(device)
                    sent2_cuda = sent2.to(device)
                    mask1_cuda = mask1.to(device)
                    mask2_cuda = mask2.to(device)
                label = label.to(torch.float32)


                if model_class == "geneformer":
                    output = model(total_emb_cuda)
                    out = torch.squeeze(m(output))
                elif model_class == "transformer":
                    # _, _, score, _  = model(sent1_cuda, sent2_cuda)
                    output = model(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda)
                    # out = torch.squeeze(m(score))
                    out = torch.squeeze(m(output))
                    
                int_out = np.around(out.detach().cpu().numpy(),0).astype(int)
                

                loss = criterion(out, label.to(device))
                loss = torch.sum(loss)

                out = out.detach().cpu()

                with torch.no_grad():
                    loss_test_record["test_loss"] += loss.item()
                    loss_test_record["test_f1"] += metrics.f1_score(label.to(torch.int), int_out, average=average_method)
                    loss_test_record["test_precision"] += metrics.precision_score(label.to(torch.int), int_out, average=average_method)
                    loss_test_record["test_recall"] += metrics.recall_score(label.to(torch.int), int_out, average=average_method)
                    loss_test_record["test_acc"] += metrics.accuracy_score(label.to(torch.int), int_out)
                    loss_test_record["test_auc"] += metrics.roc_auc_score(label, out)
                    loss_test_record["test_aupr"] += metrics.average_precision_score(label, out)


            # early stopping   
            epoch_metric = loss_test_record["test_aupr"]/len(test_loader)
            if epoch_metric > best_metric:
                best_metric = epoch_metric
                best_epoch = epoch
                best_metric_record = loss_train_record, loss_test_record
                not_improved_count = 0

                if save_model:
                    torch.save(model, model_save_path)
            else:
                not_improved_count += 1
            
            if not_improved_count > args.early_stop:
                print("Stop training, Best performance: AUPR=", best_metric, "Best epoch=", best_epoch)
                best_train_record, best_test_record = best_metric_record
                print(training_info(best_train_record, len(train_loader))+training_info(best_test_record, len(test_loader)))

                # save best training info
                if save_log:
                    if not os.path.exists(log_path):
                        create_csv(log_path, list(best_train_record.keys())+list(best_test_record.keys()))
                    with open(log_path,'a+') as f:
                        csv_write = csv.writer(f)
                        csv_write.writerow(
                            [i/len(train_loader) for i in list(best_train_record.values())]+[i/len(test_loader) for i in list(best_test_record.values())]
                            )

                if save_model:
                    print("Best model saved in", model_save_path)

                break
            

        # average loss and print
        if test_loader is not None:
            print(training_info(loss_train_record, len(train_loader))+training_info(loss_test_record, len(test_loader)))
        else:
            print(training_info(loss_train_record, len(train_loader)))

        epoch_end_time = time.time()

        print("elapsed_time\t" + str(epoch_end_time-epoch_start_time))

        epoch_start_time = epoch_end_time
    

    if test_loader is None:
        if save_model:
            torch.save(model, model_save_path)
            print("Completed Training, model saved in", model_save_path)


