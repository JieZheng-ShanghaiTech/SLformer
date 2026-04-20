import pandas as pd
import numpy as np
import math
import torch
import random
import csv
import os
from sklearn import metrics


def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_csv(path, csv_head):
                           
    with open(path,'w',newline = '',encoding='utf-8') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)

def create_dir(dir):

    if not os.path.exists(dir):
        os.makedirs(dir)


def split_data(data, test_size=0.2, cv=1, seed=1, return_idx=False):

    rdn_idx= np.random.RandomState(seed=seed).permutation(len(data))

    n_test = math.ceil(test_size * len(data))
    # n_train = len(data) - n_test

    ind_test = rdn_idx[(cv-1)*n_test:np.min([cv*n_test, len(data)])]
    ind_train = [i for i in range(len(data)) if i not in ind_test]

    data_test = data[ind_test]
    data_train = data[ind_train]

    if return_idx:
        return ind_test, ind_train
    else:
        return data_test, data_train


def split_data_by_cancer(data, test_cancer, return_idx=False, rm_dup=True):

    test_bool = [True if data[i,3]==test_cancer else False for i in range(len(data))]
    train_bool = [True if data[i,3]!=test_cancer else False for i in range(len(data))]

    data_test = data[test_bool]
    data_train = data[train_bool]

    if rm_dup:
        df_test = pd.DataFrame(data_test[:, :2], columns=['gene1', 'gene2']).drop_duplicates(keep='first')
        df_train = pd.DataFrame(data_train[:, :2], columns=['gene1', 'gene2'])

        df_combined = df_train.merge(df_test, on=['gene1', 'gene2'], how='left', indicator=True)
        df_filtered = df_combined[df_combined['_merge'] == 'left_only'].drop(columns='_merge')

        filt_train_idx = df_filtered.index
        data_train = data_train[filt_train_idx]

    if return_idx:
        ind_test = list(np.where(np.array(test_bool) == True)[0])
        ind_train = list(np.where(np.array(train_bool) == True)[0])
        return ind_test, ind_train
    else:
        return data_test, data_train


def calc_random_auc(test_loader):

    rand_auc = 0
    rand_aupr = 0

    for i, data in enumerate(test_loader):

        _, label, _, _, _ = data

        label = label.to(torch.float32)

        rand_pred = np.random.rand(len(label))
        rand_auc += metrics.roc_auc_score(label, rand_pred)
        rand_aupr += metrics.average_precision_score(label, rand_pred)

    return rand_auc/len(test_loader), rand_aupr/len(test_loader)



def get_train_test_SL(data_total, cv=1, test_size=0.2, data_all=False, split_by_cancer=False, test_cancer=None, return_idx=False):

    if data_all:
        return data_total

    if split_by_cancer==True and test_cancer is not None:
        test_data, train_data = split_data_by_cancer(data_total, test_cancer=test_cancer, return_idx=return_idx)
    else:
        test_data, train_data = split_data(data_total, test_size=test_size, cv=cv, seed=1, return_idx=return_idx)
    
    return test_data, train_data


def mean_metrics(result_fp):

    log = pd.read_csv(result_fp)
    mean = log.mean()

    return dict(zip(['avg_'+m for m in mean.index], mean.values))


def average_metrics(result_fp):

    log = pd.read_csv(result_fp)
    mean = np.round(log.mean(),4)
    std = np.round(log.std(),4)
    res = [str(mean[i])+" ("+str(std[i])+")" for i in range(len(mean))]
    with open(result_fp,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(res)


def precision_at_k(rel, pred, k):

    sorted_idx = np.argsort(pred)[::-1]
    topk_idx = sorted_idx[:k]
    rel_at_k = rel[topk_idx]
    TP = np.sum(rel_at_k)

    return TP/k


def recall_at_k(rel, pred, k):

    sorted_idx = np.argsort(pred)[::-1]
    topk_idx = sorted_idx[:k]
    rel_at_k = rel[topk_idx]
    TP = np.sum(rel_at_k)
    total_rel = np.sum(rel)

    return TP/total_rel


def hit_at_k(rel, pred, k):

    pred_topk_idx = np.argsort(pred)[::-1][:k]
    true_topk_idx = np.argsort(rel)[::-1][:k]
    hit_idx = set(pred_topk_idx.tolist()).intersection(set(true_topk_idx.tolist()))

    return len(hit_idx)


def hit_at_k_bin(rel, pred, k):

    sorted_idx = np.argsort(pred)[::-1]
    topk_idx = sorted_idx[:k]
    rel_at_k = rel[topk_idx]

    # return np.sum(rel_at_k)
    return np.sum(rel_at_k)/k


# if true labels are consecutive values
def ndcg(k, predict_list, true_list):

    hit_topk = len(set(predict_list[:k]) & set(true_list[:k]))

    # ndcg topk
    denom = np.log2(np.arange(2, k + 2))
    dcg_topk = np.sum(np.in1d(predict_list[:k], true_list[:k]) / denom)
    idcg_topk = np.sum((1 / denom)[:k])
    ndcg_topk = 0 if dcg_topk == 0 or idcg_topk == 0 else dcg_topk / idcg_topk

    return ndcg_topk, hit_topk


# if true labels are binary labels
def ndcg_bin(k, predict_list, true_list):

    hit_topk = len(set(predict_list[:k]) & set(true_list))

    # ndcg topk
    denom = np.log2(np.arange(2, k + 2))
    dcg_topk = np.sum(np.in1d(predict_list[:k], true_list) / denom)
    idcg_topk = np.sum((1 / denom)[:min(len(true_list), k)])
    ndcg_topk = 0 if dcg_topk == 0 or idcg_topk == 0 else dcg_topk / idcg_topk

    return ndcg_topk, hit_topk


def stat_indep_test(task="cancer_specific"):

    eval_cv_dir = "experiment/independent_test/eval_cv"
    if task == "cancer_specific":
        task_dir = {
            "cancer_specific": ["COAD", "LAML", "LUAD"],
        }
    elif task == "mix":
        task_dir = {
            "mix": ["mix"]
        }
    models = ["geneformer", "transformer"]
    metrics = ["ndcg@10","ndcg@20","ndcg@30","ndcg@50","ndcg@100"]

    models_perform = {m:[] for m in models}

    for task, dirs in task_dir.items():
        for dir in dirs:
            for f in os.listdir(os.path.join(eval_cv_dir, dir)):
                for model in models:
                    if model in f and "binary" not in f:
                        df = pd.read_csv(os.path.join(eval_cv_dir, dir, f))
                        
                        for i in range(len(df)):
                            for m in metrics:
                                models_perform[model].append(df.iloc[i][m])
                            
    perform_diff = np.sum(models_perform['transformer']) - np.sum(models_perform['geneformer'])
    return perform_diff