import torch
import time
import os
import logging
from sklearn import metrics
import numpy as np
import csv
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util import create_csv

import warnings
warnings.filterwarnings("ignore")


def average_epoch(loss_record, steps):

    for name, loss in loss_record.items():
        loss_record[name] = loss / steps

    return loss_record


def training_info(loss_record):

    info = ""
    for name, loss in loss_record.items():
        info += name
        info += "\t"
        info += str(np.round(loss, 4))
        info += "\t"

    return info


def train(device, model, criterion, m, args, train_loader, model_save_path, result_path, test_loader=None, save_model=False, save_result=True, model_class="geneformer", wandb_run=None):

    epoch_start_time = time.time()

    best_metric = 0 # aupr
    # best_metric = 1000 # loss
    best_epoch = 1
    not_improved_count = 0
    best_metric_record = {}, {}

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=args.eps, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam([
        {'params': model.pos_encoder.parameters(), 'lr': args.transformer_lr, 'betas': (0.9, 0.99), 'eps': args.eps, 'weight_decay': args.weight_decay},
        {'params': model.transformer_encoder.parameters(), 'lr': args.transformer_lr, 'betas': (0.9, 0.99), 'eps': args.eps, 'weight_decay': args.weight_decay},
        {'params': model.predictor.parameters(), 'lr': args.predictor_lr, 'betas': (0.9, 0.99), 'eps': args.eps, 'weight_decay': args.weight_decay},
        ])
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=True)
    #############

    device = torch.device("cuda:" + str(args.device))
    model = model.to(device)

    for epoch in range(1, args.epochs + 1):

        logging.info(f'Epoch {epoch}')

        train_record = {
            "train_loss":0,
            # "train_auc": 0,
            # "train_aupr": 0,
            # "train_f1":0,
            # "train_precision":0,
            # "train_recall":0,
            # "train_acc":0,
        }

        test_record = {
            "test_loss":0,
            # "test_auc": 0,
            # "test_aupr": 0,
            # "test_f1":0,
            # "test_precision":0,
            # "test_recall":0,
            # "test_acc":0,
        }

        # train_record = {}
        # test_record = {}

        average_method = "binary"

        pred_train = []
        pred_int_train = []
        label_train = []
        pred_test = []
        pred_int_test = []
        label_test = []

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
                # out = torch.squeeze(output)
            elif model_class == "transformer":
                output = model(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda)
                out = torch.squeeze(m(output))
                # out = torch.squeeze(output)
            
            int_out = np.around(out.detach().cpu().numpy(),0).astype(int)
            # int_out = np.argmax(out.detach().cpu(), axis=1)
            
            # loss = criterion(out, label.to(torch.long).to(device))
            loss = criterion(out, label.to(device))
            # loss = criterion(out, label.to(torch.long).to(device))
            loss = torch.sum(loss)
        
            loss.backward()
            optimizer.step()

            pred_train.append(out.detach().cpu())
            pred_int_train.append(torch.tensor(int_out))
            label_train.append(label)

            with torch.no_grad():
                train_record["train_loss"] += loss.item()
        #         train_record["train_f1"] += metrics.f1_score(label.to(torch.int), int_out, average=average_method)
        #         train_record["train_precision"] += metrics.precision_score(label.to(torch.int), int_out, average=average_method)
        #         train_record["train_recall"] += metrics.recall_score(label.to(torch.int), int_out, average=average_method)
        #         train_record["train_acc"] += metrics.accuracy_score(label.to(torch.int), int_out)
        #         train_record["train_auc"] += metrics.roc_auc_score(label, out)
        #         train_record["train_aupr"] += metrics.average_precision_score(label, out)

        train_record = average_epoch(train_record, len(train_loader))
        
        pred_train = torch.cat(pred_train, dim=0)
        pred_int_train = torch.cat(pred_int_train, dim=0)
        label_train = torch.cat(label_train, dim=0)
        with torch.no_grad():
            # train_record["train_auc"] = metrics.roc_auc_score(label_train, pred_train.softmax(dim=1)[:,1])
            # train_record["train_aupr"] = metrics.average_precision_score(label_train, pred_train.softmax(dim=1)[:, 1])
            train_record["train_auc"] = metrics.roc_auc_score(label_train, pred_train)
            train_record["train_aupr"] = metrics.average_precision_score(label_train, pred_train)
            precision, recall, _ = metrics.precision_recall_curve(label_train, pred_train)
            # train_record["train_f1"] = metrics.f1_score(label_train.to(torch.int), pred_int_train, average=average_method)
            train_record["train_f1"] = max(2 * precision * recall / (precision + recall))
            train_record["train_precision"] = metrics.precision_score(label_train.to(torch.int), pred_int_train, average=average_method)
            train_record["train_recall"] = metrics.recall_score(label_train.to(torch.int), pred_int_train, average=average_method)
            train_record["train_acc"] = metrics.accuracy_score(label_train.to(torch.int), pred_int_train)



        if test_loader is not None:

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
                    # out = torch.squeeze(output)
                elif model_class == "transformer":
                    output = model(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda)
                    out = torch.squeeze(m(output))
                    # out = torch.squeeze(output)
                    
                int_out = np.around(out.detach().cpu().numpy(),0).astype(int)
                # int_out = np.argmax(out.detach().cpu(), axis=1)

                loss = criterion(out, label.to(device))
                # loss = criterion(out, label.to(torch.long).to(device))
                loss = torch.sum(loss)

                out = out.detach().cpu()

                pred_test.append(out)
                pred_int_test.append(torch.tensor(int_out))
                label_test.append(label)

                with torch.no_grad():
                    test_record["test_loss"] += loss.item()
            #         test_record["test_f1"] += metrics.f1_score(label.to(torch.int), int_out, average=average_method)
            #         test_record["test_precision"] += metrics.precision_score(label.to(torch.int), int_out, average=average_method)
            #         test_record["test_recall"] += metrics.recall_score(label.to(torch.int), int_out, average=average_method)
            #         test_record["test_acc"] += metrics.accuracy_score(label.to(torch.int), int_out)
            #         test_record["test_auc"] += metrics.roc_auc_score(label, out)
            #         test_record["test_aupr"] += metrics.average_precision_score(label, out)

            test_record = average_epoch(test_record, len(test_loader))
            
            pred_test = torch.cat(pred_test, dim=0)
            pred_int_test = torch.cat(pred_int_test, dim=0)
            label_test = torch.cat(label_test, dim=0)
            with torch.no_grad():
                # test_record["test_auc"] = metrics.roc_auc_score(label_test, pred_test.softmax(dim=1)[:,1])
                # test_record["test_aupr"] = metrics.average_precision_score(label_test, pred_test.softmax(dim=1)[:,1])
                test_record["test_auc"] = metrics.roc_auc_score(label_test, pred_test)
                test_record["test_aupr"] = metrics.average_precision_score(label_test, pred_test)
                precision, recall, _ = metrics.precision_recall_curve(label_test, pred_test)
                # test_record["test_f1"] = metrics.f1_score(label_test.to(torch.int), pred_int_test, average=average_method)
                test_record["test_f1"] = max(2 * precision * recall / (precision + recall))
                test_record["test_precision"] = metrics.precision_score(label_test.to(torch.int), pred_int_test, average=average_method)
                test_record["test_recall"] = metrics.recall_score(label_test.to(torch.int), pred_int_test, average=average_method)
                test_record["test_acc"] = metrics.accuracy_score(label_test.to(torch.int), pred_int_test)
            
            # scheduler.step(test_record["test_aupr"])
            scheduler.step(test_record["test_aupr"]) #############

            # early stopping   
            epoch_metric = test_record["test_aupr"]
            # epoch_metric = test_record["test_loss"]
            # epoch_metric = test_record["test_auc"]
            if epoch_metric > best_metric: # aupr
            # if epoch_metric < best_metric: # loss
                best_metric = epoch_metric
                best_epoch = epoch
                best_metric_record = train_record, test_record
                not_improved_count = 0

                if save_model:
                    torch.save(model.state_dict(), model_save_path)
            else:
                not_improved_count += 1
            
            if not_improved_count > args.early_stop:
                logging.info("Stop training, Best performance: AUPR= "+str(best_metric)+" Best epoch= "+str(best_epoch))
                best_train_record, best_test_record = best_metric_record
                logging.info(training_info(best_train_record)+training_info(best_test_record))

                # save best training performance
                if save_result:
                    if not os.path.exists(result_path):
                        create_csv(result_path, list(best_train_record.keys())+list(best_test_record.keys()))
                    with open(result_path,'a+') as f:
                        csv_write = csv.writer(f)
                        csv_write.writerow(list(best_train_record.values())+list(best_test_record.values()))
                        # csv_write.writerow(
                        #     [i/len(train_loader) for i in list(best_train_record.values())]+[i/len(test_loader) for i in list(best_test_record.values())]
                        #     )

                if save_model:
                    logging.info("Best model saved in "+model_save_path)

                ## log wandb
                if wandb_run is not None:
                    wandb_run.log(best_train_record)
                    wandb_run.log(best_test_record)

                break
            

        # average loss and print
        if test_loader is not None:
            logging.info(training_info(train_record)+training_info(test_record))
        else:
            logging.info(training_info(train_record))

        epoch_end_time = time.time()

        logging.info("elapsed_time\t" + str(epoch_end_time-epoch_start_time))

        epoch_start_time = epoch_end_time
    

    if test_loader is None:
        if save_model:
            torch.save(model.state_dict(), model_save_path)
            logging.info("Completed Training, model saved in "+model_save_path)


def train_latefusion(device, model, criterion, m, args, train_loader, model_save_path, result_path, test_loader=None, save_model=False, save_result=True, model_class="geneformer", wandb_run=None):

    epoch_start_time = time.time()

    best_metric = 0 # aupr
    # best_metric = 1000 # loss
    best_epoch = 1
    not_improved_count = 0
    best_metric_record = {}, {}

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=args.eps, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam([
        {'params': model.pos_encoder.parameters(), 'lr': args.transformer_lr, 'betas': (0.9, 0.99), 'eps': args.eps, 'weight_decay': args.weight_decay},
        {'params': model.transformer_encoder.parameters(), 'lr': args.transformer_lr, 'betas': (0.9, 0.99), 'eps': args.eps, 'weight_decay': args.weight_decay},
        {'params': model.predictor.parameters(), 'lr': args.predictor_lr, 'betas': (0.9, 0.99), 'eps': args.eps, 'weight_decay': args.weight_decay},
        ])
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=True)
    #############

    device = torch.device("cuda:" + str(args.device))
    model = model.to(device)

    for epoch in range(1, args.epochs + 1):

        logging.info(f'Epoch {epoch}')

        train_record = {
            "train_loss":0,
            # "train_auc": 0,
            # "train_aupr": 0,
            # "train_f1":0,
            # "train_precision":0,
            # "train_recall":0,
            # "train_acc":0,
        }

        test_record = {
            "test_loss":0,
            # "test_auc": 0,
            # "test_aupr": 0,
            # "test_f1":0,
            # "test_precision":0,
            # "test_recall":0,
            # "test_acc":0,
        }

        # train_record = {}
        # test_record = {}

        average_method = "binary"

        pred_train = []
        pred_int_train = []
        label_train = []
        pred_test = []
        pred_int_test = []
        label_test = []

        model.train()

        for i, data in enumerate(train_loader):

            if model_class == "geneformer":
                total_emb, label, _, _, _ = data
                total_emb_cuda = torch.autograd.Variable(total_emb.to(device)).to(torch.float32)
            elif model_class == "transformer":
                # sent1, sent2, label, _, _ = data
                sent1, mask1, sent2, mask2, label, g1id, g2id, _ = data
                sent1_cuda = sent1.to(device)
                sent2_cuda = sent2.to(device)
                mask1_cuda = mask1.to(device)
                mask2_cuda = mask2.to(device)
                g1id_cuda = g1id.to(device)
                g2id_cuda = g2id.to(device)
            label = label.to(torch.float32)

            optimizer.zero_grad()

            if model_class == "geneformer":
                output = model(total_emb_cuda)
                out = torch.squeeze(m(output))
                # out = torch.squeeze(output)
            elif model_class == "transformer":
                output = model(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda, g1id_cuda, g2id_cuda, device)
                out = torch.squeeze(m(output))
                # out = torch.squeeze(output)
            
            int_out = np.around(out.detach().cpu().numpy(),0).astype(int)
            # int_out = np.argmax(out.detach().cpu(), axis=1)
            
            # loss = criterion(out, label.to(torch.long).to(device))
            loss = criterion(out, label.to(device))
            # loss = criterion(out, label.to(torch.long).to(device))
            loss = torch.sum(loss)
        
            loss.backward()
            optimizer.step()

            pred_train.append(out.detach().cpu())
            pred_int_train.append(torch.tensor(int_out))
            label_train.append(label)

            with torch.no_grad():
                train_record["train_loss"] += loss.item()
        #         train_record["train_f1"] += metrics.f1_score(label.to(torch.int), int_out, average=average_method)
        #         train_record["train_precision"] += metrics.precision_score(label.to(torch.int), int_out, average=average_method)
        #         train_record["train_recall"] += metrics.recall_score(label.to(torch.int), int_out, average=average_method)
        #         train_record["train_acc"] += metrics.accuracy_score(label.to(torch.int), int_out)
        #         train_record["train_auc"] += metrics.roc_auc_score(label, out)
        #         train_record["train_aupr"] += metrics.average_precision_score(label, out)

        train_record = average_epoch(train_record, len(train_loader))
        
        pred_train = torch.cat(pred_train, dim=0)
        pred_int_train = torch.cat(pred_int_train, dim=0)
        label_train = torch.cat(label_train, dim=0)
        with torch.no_grad():
            # train_record["train_auc"] = metrics.roc_auc_score(label_train, pred_train.softmax(dim=1)[:,1])
            # train_record["train_aupr"] = metrics.average_precision_score(label_train, pred_train.softmax(dim=1)[:, 1])
            train_record["train_auc"] = metrics.roc_auc_score(label_train, pred_train)
            train_record["train_aupr"] = metrics.average_precision_score(label_train, pred_train)
            precision, recall, _ = metrics.precision_recall_curve(label_train, pred_train)
            # train_record["train_f1"] = metrics.f1_score(label_train.to(torch.int), pred_int_train, average=average_method)
            train_record["train_f1"] = max(2 * precision * recall / (precision + recall))
            train_record["train_precision"] = metrics.precision_score(label_train.to(torch.int), pred_int_train, average=average_method)
            train_record["train_recall"] = metrics.recall_score(label_train.to(torch.int), pred_int_train, average=average_method)
            train_record["train_acc"] = metrics.accuracy_score(label_train.to(torch.int), pred_int_train)



        if test_loader is not None:

            model.eval()

            for i, data in enumerate(test_loader):
            
                if model_class == "geneformer":
                    total_emb, label, _, _, _ = data
                    total_emb_cuda = torch.autograd.Variable(total_emb.to(device)).to(torch.float32)
                elif model_class == "transformer":
                    # sent1, sent2, label, _, _ = data
                    sent1, mask1, sent2, mask2, label, g1id, g2id, _ = data
                    sent1_cuda = sent1.to(device)
                    sent2_cuda = sent2.to(device)
                    mask1_cuda = mask1.to(device)
                    mask2_cuda = mask2.to(device)
                    g1id_cuda = g1id.to(device)
                    g2id_cuda = g2id.to(device)
                label = label.to(torch.float32)


                if model_class == "geneformer":
                    output = model(total_emb_cuda)
                    out = torch.squeeze(m(output))
                    # out = torch.squeeze(output)
                elif model_class == "transformer":
                    output = model(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda, g1id_cuda, g2id_cuda, device)
                    out = torch.squeeze(m(output))
                    # out = torch.squeeze(output)
                    
                int_out = np.around(out.detach().cpu().numpy(),0).astype(int)
                # int_out = np.argmax(out.detach().cpu(), axis=1)

                loss = criterion(out, label.to(device))
                # loss = criterion(out, label.to(torch.long).to(device))
                loss = torch.sum(loss)

                out = out.detach().cpu()

                pred_test.append(out)
                pred_int_test.append(torch.tensor(int_out))
                label_test.append(label)

                with torch.no_grad():
                    test_record["test_loss"] += loss.item()
            #         test_record["test_f1"] += metrics.f1_score(label.to(torch.int), int_out, average=average_method)
            #         test_record["test_precision"] += metrics.precision_score(label.to(torch.int), int_out, average=average_method)
            #         test_record["test_recall"] += metrics.recall_score(label.to(torch.int), int_out, average=average_method)
            #         test_record["test_acc"] += metrics.accuracy_score(label.to(torch.int), int_out)
            #         test_record["test_auc"] += metrics.roc_auc_score(label, out)
            #         test_record["test_aupr"] += metrics.average_precision_score(label, out)

            test_record = average_epoch(test_record, len(test_loader))
            
            pred_test = torch.cat(pred_test, dim=0)
            pred_int_test = torch.cat(pred_int_test, dim=0)
            label_test = torch.cat(label_test, dim=0)
            with torch.no_grad():
                # test_record["test_auc"] = metrics.roc_auc_score(label_test, pred_test.softmax(dim=1)[:,1])
                # test_record["test_aupr"] = metrics.average_precision_score(label_test, pred_test.softmax(dim=1)[:,1])
                test_record["test_auc"] = metrics.roc_auc_score(label_test, pred_test)
                test_record["test_aupr"] = metrics.average_precision_score(label_test, pred_test)
                precision, recall, _ = metrics.precision_recall_curve(label_test, pred_test)
                # test_record["test_f1"] = metrics.f1_score(label_test.to(torch.int), pred_int_test, average=average_method)
                test_record["test_f1"] = max(2 * precision * recall / (precision + recall))
                test_record["test_precision"] = metrics.precision_score(label_test.to(torch.int), pred_int_test, average=average_method)
                test_record["test_recall"] = metrics.recall_score(label_test.to(torch.int), pred_int_test, average=average_method)
                test_record["test_acc"] = metrics.accuracy_score(label_test.to(torch.int), pred_int_test)
            
            # scheduler.step(test_record["test_aupr"])
            scheduler.step(test_record["test_aupr"]) #############

            # early stopping   
            epoch_metric = test_record["test_aupr"]
            # epoch_metric = test_record["test_loss"]
            # epoch_metric = test_record["test_auc"]
            if epoch_metric > best_metric: # aupr
            # if epoch_metric < best_metric: # loss
                best_metric = epoch_metric
                best_epoch = epoch
                best_metric_record = train_record, test_record
                not_improved_count = 0

                if save_model:
                    torch.save(model.state_dict(), model_save_path)
            else:
                not_improved_count += 1
            
            if not_improved_count > args.early_stop:
                logging.info("Stop training, Best performance: AUPR= "+str(best_metric)+" Best epoch= "+str(best_epoch))
                best_train_record, best_test_record = best_metric_record
                logging.info(training_info(best_train_record)+training_info(best_test_record))

                # save best training performance
                if save_result:
                    if not os.path.exists(result_path):
                        create_csv(result_path, list(best_train_record.keys())+list(best_test_record.keys()))
                    with open(result_path,'a+') as f:
                        csv_write = csv.writer(f)
                        csv_write.writerow(list(best_train_record.values())+list(best_test_record.values()))
                        # csv_write.writerow(
                        #     [i/len(train_loader) for i in list(best_train_record.values())]+[i/len(test_loader) for i in list(best_test_record.values())]
                        #     )

                if save_model:
                    logging.info("Best model saved in "+model_save_path)

                ## log wandb
                if wandb_run is not None:
                    wandb_run.log(best_train_record)
                    wandb_run.log(best_test_record)

                break
            

        # average loss and print
        if test_loader is not None:
            logging.info(training_info(train_record)+training_info(test_record))
        else:
            logging.info(training_info(train_record))

        epoch_end_time = time.time()

        logging.info("elapsed_time\t" + str(epoch_end_time-epoch_start_time))

        epoch_start_time = epoch_end_time
    

    if test_loader is None:
        if save_model:
            torch.save(model.state_dict(), model_save_path)
            logging.info("Completed Training, model saved in "+model_save_path)



# def pretrain(device, model, criterion, args, train_loader, test_loader, model_save_path, result_path, save_model=False, save_result=True):

#     epoch_start_time = time.time()

#     best_metric = np.inf
#     best_epoch = 1
#     not_improved_count = 0
#     best_metric_record = {}, {}

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=args.eps, weight_decay=args.weight_decay)
#     # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience)

#     device = torch.device("cuda:" + str(args.device))
#     # device = torch.device("cpu")

#     model = model.to(device)

#     for epoch in range(1, args.epochs + 1):

#         logging.info(f'Epoch {epoch}')

#         train_record = {
#             "train_loss":0,
#         }
#         test_record = {
#             "test_loss":0,
#         }


#         average_method = "macro"

#         pred_train = []
#         pred_int_train = []
#         label_train = []
#         pred_test = []
#         pred_int_test = []
#         label_test = []

#         model.train()

#         for i, data in enumerate(train_loader):

#             sent, att_mask, root_gene, _ = data
#             sent_cuda = sent.to(device)
#             mask_cuda = att_mask.to(device)
#             label = root_gene.to(torch.float32)

#             optimizer.zero_grad()

#             output = model(sent_cuda, mask_cuda)
#             out = torch.squeeze(output)
            
#             int_out = np.argmax(out.detach().cpu(), axis=1)
            
#             loss = criterion(out, label.to(torch.long).to(device))
#             loss = torch.sum(loss)
        
#             loss.backward()
#             optimizer.step()

#             pred_train.append(out.detach().cpu())
#             pred_int_train.append(torch.tensor(int_out))
#             label_train.append(label)

#             with torch.no_grad():
#                 train_record["train_loss"] += loss.item()

#         train_record = average_epoch(train_record, len(train_loader))
        
#         pred_train = torch.cat(pred_train, dim=0)
#         pred_int_train = torch.cat(pred_int_train, dim=0)
#         label_train = torch.cat(label_train, dim=0)
#         with torch.no_grad():
#             # precision, recall, _ = metrics.precision_recall_curve(label_train, pred_train)
#             train_record["train_f1"] = metrics.f1_score(label_train.to(torch.int), pred_int_train, average=average_method)
#             # train_record["train_f1"] = max(2 * precision * recall / (precision + recall))
#             train_record["train_precision"] = metrics.precision_score(label_train.to(torch.int), pred_int_train, average=average_method)
#             train_record["train_recall"] = metrics.recall_score(label_train.to(torch.int), pred_int_train, average=average_method)
#             train_record["train_acc"] = metrics.accuracy_score(label_train.to(torch.int), pred_int_train)



#         if test_loader is not None:

#             model.eval()

#             for i, data in enumerate(test_loader):
            
#                 sent, att_mask, root_gene, _ = data
#                 sent_cuda = sent.to(device)
#                 mask_cuda = att_mask.to(device)
#                 label = root_gene.to(torch.float32)


#                 output = model(sent_cuda, mask_cuda)
#                 out = torch.squeeze(output)
                    
#                 int_out = np.argmax(out.detach().cpu(), axis=1)

#                 loss = criterion(out, label.to(torch.long).to(device))
#                 loss = torch.sum(loss)

#                 out = out.detach().cpu()

#                 pred_test.append(out)
#                 pred_int_test.append(torch.tensor(int_out))
#                 label_test.append(label)

#                 with torch.no_grad():
#                     test_record["test_loss"] += loss.item()

#             test_record = average_epoch(test_record, len(test_loader))
            
#             pred_test = torch.cat(pred_test, dim=0)
#             pred_int_test = torch.cat(pred_int_test, dim=0)
#             label_test = torch.cat(label_test, dim=0)
#             with torch.no_grad():
#                 # precision, recall, _ = metrics.precision_recall_curve(label_test, pred_test)
#                 test_record["test_f1"] = metrics.f1_score(label_test.to(torch.int), pred_int_test, average=average_method)
#                 # test_record["test_f1"] = max(2 * precision * recall / (precision + recall))
#                 test_record["test_precision"] = metrics.precision_score(label_test.to(torch.int), pred_int_test, average=average_method)
#                 test_record["test_recall"] = metrics.recall_score(label_test.to(torch.int), pred_int_test, average=average_method)
#                 test_record["test_acc"] = metrics.accuracy_score(label_test.to(torch.int), pred_int_test)
            
#             # scheduler.step(test_record["test_aupr"])

#             # early stopping   
#             epoch_metric = test_record["test_loss"]
#             if epoch_metric < best_metric:
#                 best_metric = epoch_metric
#                 best_epoch = epoch
#                 best_metric_record = train_record, test_record
#                 not_improved_count = 0

#                 if save_model:
#                     torch.save(model.state_dict(), model_save_path)
#             else:
#                 not_improved_count += 1
            
#             if not_improved_count > args.early_stop:
#                 best_train_record, best_test_record = best_metric_record
#                 logging.info("Stop training, Best performance: F1= "+str(best_test_record['test_f1'])+" Best epoch= "+str(best_epoch))
#                 logging.info(training_info(best_train_record)+training_info(best_test_record))

#                 # save best training performance
#                 if save_result:
#                     if not os.path.exists(result_path):
#                         create_csv(result_path, list(best_train_record.keys())+list(best_test_record.keys()))
#                     with open(result_path,'a+') as f:
#                         csv_write = csv.writer(f)
#                         csv_write.writerow(list(best_train_record.values())+list(best_test_record.values()))
#                         # csv_write.writerow(
#                         #     [i/len(train_loader) for i in list(best_train_record.values())]+[i/len(test_loader) for i in list(best_test_record.values())]
#                         #     )

#                 if save_model:
#                     logging.info("Best model saved in "+model_save_path)

#                 break
            

#         logging.info(training_info(train_record)+training_info(test_record))

#         epoch_end_time = time.time()

#         logging.info("elapsed_time\t" + str(epoch_end_time-epoch_start_time))

#         epoch_start_time = epoch_end_time


def pretrain(device, model, criterion, args, data_loader, model_save_path, result_path, save_model=False, save_result=True):

    epoch_start_time = time.time()

    best_metric = np.inf
    best_epoch = 12
    not_improved_count = 0
    loss_thr = 0.01
    best_metric_record = {}, {}

    optimizer = torch.optim.Adam([
        {'params': model.pos_encoder.parameters(), 'lr': args.transformer_lr, 'betas': (0.9, 0.99), 'eps': args.eps, 'weight_decay': args.weight_decay},
        {'params': model.transformer_encoder.parameters(), 'lr': args.transformer_lr, 'betas': (0.9, 0.99), 'eps': args.eps, 'weight_decay': args.weight_decay},
        {'params': model.predictor.parameters(), 'lr': args.predictor_lr, 'betas': (0.9, 0.99), 'eps': args.eps, 'weight_decay': args.weight_decay},
        ])
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience)

    device = torch.device("cuda:" + str(args.device))
    # device = torch.device("cpu")

    model = model.to(device)

    for epoch in range(1, args.epochs + 1):

        logging.info(f'Epoch {epoch}')

        train_record = {
            "train_loss":0,
        }


        average_method = "macro"

        pred_train = []
        pred_int_train = []
        label_train = []

        model.train()

        for i, data in enumerate(data_loader):

            sent, att_mask, root_gene, _ = data
            sent_cuda = sent.to(device)
            mask_cuda = att_mask.to(device)
            label = root_gene.to(torch.float32)

            optimizer.zero_grad()

            output = model(sent_cuda, mask_cuda)
            out = torch.squeeze(output)
            
            int_out = np.argmax(out.detach().cpu(), axis=1)
            
            loss = criterion(out, label.to(torch.long).to(device))
            loss = torch.sum(loss)
        
            loss.backward()
            optimizer.step()

            pred_train.append(out.detach().cpu())
            pred_int_train.append(torch.tensor(int_out))
            label_train.append(label)

            with torch.no_grad():
                train_record["train_loss"] += loss.item()

        train_record = average_epoch(train_record, len(data_loader))
        
        pred_train = torch.cat(pred_train, dim=0)
        pred_int_train = torch.cat(pred_int_train, dim=0)
        label_train = torch.cat(label_train, dim=0)
        with torch.no_grad():
            # precision, recall, _ = metrics.precision_recall_curve(label_train, pred_train)
            train_record["train_f1"] = metrics.f1_score(label_train.to(torch.int), pred_int_train, average=average_method)
            # train_record["train_f1"] = max(2 * precision * recall / (precision + recall))
            train_record["train_precision"] = metrics.precision_score(label_train.to(torch.int), pred_int_train, average=average_method)
            train_record["train_recall"] = metrics.recall_score(label_train.to(torch.int), pred_int_train, average=average_method)
            train_record["train_acc"] = metrics.accuracy_score(label_train.to(torch.int), pred_int_train)


            # early stopping   
            epoch_metric = train_record["train_loss"]
            if best_metric - epoch_metric > loss_thr:
                best_metric = epoch_metric
                best_epoch = epoch
                best_metric_record = train_record
                not_improved_count = 0

                if save_model:
                    torch.save(model.state_dict(), model_save_path)
            else:
                not_improved_count += 1
            
            if not_improved_count > args.early_stop:
                best_train_record = best_metric_record
                logging.info("Stop training, Best performance: F1= "+str(best_train_record['train_f1'])+" Best epoch= "+str(best_epoch))
                logging.info(training_info(best_train_record))

                # save best training performance
                if save_result:
                    if not os.path.exists(result_path):
                        create_csv(result_path, list(best_train_record.keys()))
                    with open(result_path,'a+') as f:
                        csv_write = csv.writer(f)
                        csv_write.writerow(list(best_train_record.values()))
                        # csv_write.writerow(
                        #     [i/len(train_loader) for i in list(best_train_record.values())]+[i/len(test_loader) for i in list(best_test_record.values())]
                        #     )

                if save_model:
                    logging.info("Best model saved in "+model_save_path)

                break
            

        logging.info(training_info(train_record))

        epoch_end_time = time.time()
        logging.info("elapsed_time\t" + str(epoch_end_time-epoch_start_time))
        epoch_start_time = epoch_end_time