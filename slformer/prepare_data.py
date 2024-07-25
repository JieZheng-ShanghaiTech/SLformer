import argparse
import yaml
import easydict
import numpy as np
import os
import json
from sklearn.metrics import average_precision_score, precision_recall_curve

from preprocess import Data_Preprocess
from dataloader import prepare_SL_data
from task import Validation_Experiment
from util import create_dir, get_train_test_SL, transfer_data_idx, find_data_idx


def save_train_test_data(config, common_data):

    cancer_list = common_data["cancer_list"]
    id2cancer_map = {i:c for c,i in common_data["cancer2id_map"].items()}

    ## cancer-specific and mix
    save_dir = os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cancer_specific_all")
    create_dir(save_dir)
    data_mix_train = {i:[] for i in range(1,6)}
    data_mix_test = {i:[] for i in range(1,6)}
    for cancer_type in cancer_list:
        data_total = prepare_SL_data(
            config=config,
            cancer=cancer_type,
            common_data=common_data,
            type="general"
        )
        for cv in range(1,6):
            data_test, data_train = get_train_test_SL(data_total, cv=cv, data_all=False, return_idx=False)
            np.save(os.path.join(save_dir, f"test_{cancer_type}_fold_{cv}.npy"), data_test)
            np.save(os.path.join(save_dir, f"train_{cancer_type}_fold_{cv}.npy"), data_train)
            data_mix_train[cv].append(data_train)
            data_mix_test[cv].append(data_test)

    ## mix
    save_dir = os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "mix_all")
    create_dir(save_dir)
    for cv in range(1,6):
        data_train = np.concatenate(data_mix_train[cv], axis=0)
        data_test = np.concatenate(data_mix_test[cv], axis=0)
        np.save(os.path.join(save_dir, f"test_all_fold_{cv}.npy"), data_test)
        np.save(os.path.join(save_dir, f"train_all_fold_{cv}.npy"), data_train)

    # for cv in range(1,6):
        # data_test, data_train = get_train_test_SL(data_total, cv=cv, data_all=False, return_idx=False)
        # np.save(os.path.join(save_dir, f"test_all_fold_{cv}.npy"), data_test)
        # np.save(os.path.join(save_dir, f"train_all_fold_{cv}.npy"), data_train)

            
    ## cross-cancer
    save_dir = os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cross_cancer_all")
    create_dir(save_dir)
    data_total = prepare_SL_data(
        config=config,
        cancer='all',
        common_data=common_data,
        type="general"
    )
    for cancer in range(len(cancer_list)):
        cancer_name = id2cancer_map[cancer]
    # for cancer, cancer_name in id2cancer_map.items():
        data_test, data_train = get_train_test_SL(data_total, data_all=False, split_by_cancer=True, test_cancer=cancer, return_idx=False)
        np.save(os.path.join(save_dir, f"test_{cancer_name}.npy"), data_test)
        np.save(os.path.join(save_dir, f"train_{cancer_name}.npy"), data_train)
            


def save_independent_test_data(config, data_total):

    save_dir = os.path.join(config.SAVED_DATA_DIR, "independent_test_data")

    for data_type, data in data_total.items():
        np.save(os.path.join(save_dir, f"{data_type}.npy"), data)



def get_benchmark_df(config, common_data):

    cancer_list = common_data["cancer_list"]
    gene2id_map = common_data["gene2id_map"]
    id2cancer_map = {i:c for c,i in common_data["cancer2id_map"].items()}
    save_dir = os.path.join(config.SAVED_DATA_DIR, "benchmark_data")
    create_dir(os.path.join(save_dir, "cancer_specific"))
    create_dir(os.path.join(save_dir, "mix"))
    create_dir(os.path.join(save_dir, "cross_cancer"))

    # cancer_specific
    for cancer_type in cancer_list:
        for cv in range(1,6):
            data_test = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cancer_specific_all", f"test_{cancer_type}_fold_{cv}.npy"))
            data_train = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cancer_specific_all", f"train_{cancer_type}_fold_{cv}.npy"))
            df_test = transfer_data_idx(data_test, gene2id_map, id2cancer_map)
            df_train = transfer_data_idx(data_train, gene2id_map, id2cancer_map)
            df_test.to_csv(os.path.join(save_dir, "cancer_specific_all", f"test_{cancer_type}_fold_{cv}.csv"), index=False)
            df_train.to_csv(os.path.join(save_dir, "cancer_specific_all", f"train_{cancer_type}_fold_{cv}.csv"), index=False)
    # mix
    for cv in range(1,6):
        data_test = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "mix_all", f"test_all_fold_{cv}.npy"))
        data_train = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "mix_all", f"train_all_fold_{cv}.npy"))
        df_test = transfer_data_idx(data_test, gene2id_map, id2cancer_map)
        df_train = transfer_data_idx(data_train, gene2id_map, id2cancer_map)
        df_test.to_csv(os.path.join(save_dir, "mix_all", f"test_all_fold_{cv}.csv"), index=False)
        df_train.to_csv(os.path.join(save_dir, "mix_all", f"train_all_fold_{cv}.csv"), index=False)
    # cross_cancer
    for cancer in range(len(cancer_list)):
        cancer_name = id2cancer_map[cancer]
    # for cancer, cancer_name in id2cancer_map.items():
        data_test = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cross_cancer_all", f"test_{cancer_name}.npy"))
        data_train = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cross_cancer_all", f"train_{cancer_name}.npy"))
        df_test = transfer_data_idx(data_test, gene2id_map, id2cancer_map)
        df_train = transfer_data_idx(data_train, gene2id_map, id2cancer_map)
        df_test.to_csv(os.path.join(save_dir, "cross_cancer_all", f"test_{cancer_name}.csv"), index=False)
        df_train.to_csv(os.path.join(save_dir, "cross_cancer_all", f"train_{cancer_name}.csv"), index=False)


def prepare_ELISL_data(config, common_data, ELISL_data_dir):

    cancer_list = common_data["cancer_list"]
    gene2id_map = common_data["gene2id_map"]
    cancer2id_map = common_data["cancer2id_map"]
    id2cancer_map = {i:c for c,i in cancer2id_map.items()}

    df_dir = os.path.join(ELISL_data_dir, "df")
    idx_dir = os.path.join(ELISL_data_dir, "idx")
    create_dir(df_dir)
    create_dir(idx_dir)

    data_test_all = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "mix", f"test_all_fold_1.npy"))
    data_train_all = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "mix", f"train_all_fold_1.npy"))
    data_all = np.concatenate((data_train_all, data_test_all), axis=0)
    df_all = transfer_data_idx(data_all, gene2id_map, id2cancer_map, label_name='class')
    df_all.to_csv(os.path.join(df_dir, "SL_all.csv"), index=False)

    # 'cancer-specific' and 'mix'
    for cancer_type in cancer_list+['all']:
        if cancer_type == 'all':
            experiment = 'mix'
            data_cancer = data_all
        else:
            experiment = 'cancer_specific'
            data_cancer = data_all[data_all[:,-1] == cancer2id_map[cancer_type]]

        fold_idx = {}
        
        for cv in range(1,6):
            data_test = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", experiment, f"test_{cancer_type}_fold_{cv}.npy"))
            data_train = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", experiment, f"train_{cancer_type}_fold_{cv}.npy"))
            test_idx = find_data_idx(data_test, data_cancer)
            train_idx = find_data_idx(data_train, data_cancer)

            fold_idx[str(cv-1)] = {"train":list(train_idx), "test":list(test_idx)}
        
        with open(os.path.join(idx_dir, f"{cancer_type}_idx.json"), 'w') as fp:
            json.dump(fold_idx, fp)

    # 'cross-cancer'
    for cancer in range(len(cancer_list)):
        cancer_name = id2cancer_map[cancer]
    # for cancer, cancer_name in id2cancer_map.items():

        fold_idx = {}

        data_test = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cross_cancer", f"test_{cancer_name}.npy"))
        data_train = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cross_cancer", f"train_{cancer_name}.npy"))
        test_idx = find_data_idx(data_test, data_all)
        train_idx = find_data_idx(data_train, data_all)

        fold_idx[str(0)] = {"train":list(train_idx), "test":list(test_idx)}
    
        with open(os.path.join(idx_dir, f"{cancer_name}_transfer_idx.json"), 'w') as fp:
            json.dump(fold_idx, fp)



def filt_SL_test(gene_candidates, gene2id_map, gene_emb_map, geneformer_emb_map, context=8):

    candidate_filt = list(set(gene_candidates).intersection(set(gene2id_map.keys())))
    candidate_id = [gene2id_map[g] for g in candidate_filt]

    gene_context = list(gene_emb_map[context].keys())
    gene_context_geneformer = list(geneformer_emb_map[context].keys())
    gene_context_overlap = list(set(gene_context).intersection(set(gene_context_geneformer)))
    candidate_id_filt = list(set(candidate_id).intersection(set(gene_context_overlap)))

    return candidate_id_filt


def prepare_IDH1_DDR_data(common_data, cancer="Glioma"):

    DDR_genelist_dir = "./data/DDR_genelist"
    save_dir = "./data/saved_data/inference"
    create_dir(save_dir)

    gene_sent_map = common_data["gene2sent_map"]
    geneformer_emb_map=common_data["geneformer_emb_map"]
    gene2id_map = common_data["gene2id_map"]
    cancer2id_map = common_data["cancer2id_map"]

    with open(os.path.join(DDR_genelist_dir, "kegg_dna_replication.txt")) as f:
        kegg_genelist = [line.rstrip('\n') for line in f]
    with open(os.path.join(DDR_genelist_dir, "reactome_dna_repair.txt")) as f:
        reactome_genelist = [line.rstrip('\n') for line in f]

    cancer_id = cancer2id_map[cancer]
    gene_reactome_filt = filt_SL_test(reactome_genelist, gene2id_map, gene_sent_map, geneformer_emb_map, context=cancer_id)
    gene_kegg_filt = filt_SL_test(kegg_genelist, gene2id_map, gene_sent_map, geneformer_emb_map, context=cancer_id)

    data_reactome = []
    for g in gene_reactome_filt:
        data_reactome.append([gene2id_map['IDH1'], g, 0, cancer_id])

    data_reactome.append([gene2id_map['IDH1'], gene2id_map['BCL2L1'], 0, cancer_id])
    data_reactome.append([gene2id_map['IDH1'], gene2id_map['NUDT1'], 0, cancer_id])

    data_kegg = []
    for g in gene_kegg_filt:
        data_kegg.append([gene2id_map['IDH1'], g, 0, cancer_id])

    np.save(os.path.join(save_dir, f"IDH1_DDR_{cancer}_reactome.npy"), np.array(data_reactome))
    np.save(os.path.join(save_dir, f"IDH1_DDR_{cancer}_kegg.npy"), np.array(data_kegg))



def get_random_aupr_f1(config, common_data):

    np.random.seed(1)

    cancer_list = common_data["cancer_list"]
    id2cancer_map = {i:c for c,i in common_data["cancer2id_map"].items()}

    print("Calculating random AUPR and F1...")

    ## cancer-specific

    for cancer_type in cancer_list:
        rand_aupr_res = []
        rand_f1_res = []
        for cv in range(1,6):
            data_test = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cancer_specific", f"test_{cancer_type}_fold_{cv}.npy"))
            labels = data_test[:,2]
            rand_pred = np.random.rand(len(labels))
            rand_aupr_res.append(average_precision_score(labels, rand_pred))
            precision, recall, _ = precision_recall_curve(labels, rand_pred)
            rand_f1_res.append(max(2 * precision * recall / (precision + recall)))

        aupr_mean = np.round(np.mean(rand_aupr_res),4)
        aupr_std = np.round(np.std(rand_aupr_res),4)
        print(cancer_type, "cancer-specific AUPR", str(aupr_mean)+" ("+str(aupr_std)+")")
        f1_mean = np.round(np.mean(rand_f1_res),4)
        f1_std = np.round(np.std(rand_f1_res),4)
        print(cancer_type, "cancer-specific F1", str(f1_mean)+" ("+str(f1_std)+")")

    ## mix
    rand_aupr_res = []
    rand_f1_res = []
    for cv in range(1,6):
        data_test = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "mix_all", f"test_all_fold_{cv}.npy"))
        labels = data_test[:,2]
        rand_pred = np.random.rand(len(labels))
        rand_aupr_res.append(average_precision_score(labels, rand_pred))
        precision, recall, _ = precision_recall_curve(labels, rand_pred)
        rand_f1_res.append(max(2 * precision * recall / (precision + recall)))

    aupr_mean = np.round(np.mean(rand_aupr_res),4)
    aupr_std = np.round(np.std(rand_aupr_res),4)
    print("mixed cancer types AUPR", str(aupr_mean)+" ("+str(aupr_std)+")")
    f1_mean = np.round(np.mean(rand_f1_res),4)
    f1_std = np.round(np.std(rand_f1_res),4)
    print("mixed cancer types F1", str(f1_mean)+" ("+str(f1_std)+")")

    ## cross-cancer
    rand_aupr_res = []
    rand_f1_res = []
    for cancer in range(len(cancer_list)):
        cancer_name = id2cancer_map[cancer]
    # for cancer, cancer_name in id2cancer_map.items():
        data_test = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cross_cancer", f"test_{cancer_name}.npy"))
        labels = data_test[:,2]
        rand_pred = np.random.rand(len(labels))
        rand_aupr_res.append(average_precision_score(labels, rand_pred))
        precision, recall, _ = precision_recall_curve(labels, rand_pred)
        rand_f1_res.append(max(2 * precision * recall / (precision + recall)))

    aupr_mean = np.round(np.mean(rand_aupr_res),4)
    aupr_std = np.round(np.std(rand_aupr_res),4)
    print("cross cancer AUPR", str(aupr_mean)+" ("+str(aupr_std)+")")
    f1_mean = np.round(np.mean(rand_f1_res),4)
    f1_std = np.round(np.std(rand_f1_res),4)
    print("cross cancer F1", str(f1_mean)+" ("+str(f1_std)+")")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='prepare SL data')

    parser.add_argument('--data_config_file', type=str, default="./config/data_preprocess.yaml",
                    help='data preprocess config file path')
    parser.add_argument('--config_file', type=str, default="./config/independent_test.yaml",
                        help='config file path')
    args = parser.parse_args()


    with open(args.data_config_file, 'r') as f:
        data_config = easydict.EasyDict(yaml.safe_load(f))
    with open(args.config_file, 'r') as f:
        config = easydict.EasyDict(yaml.safe_load(f))

    data_preprocess = Data_Preprocess(data_config)
    common_data = data_preprocess.get_common_data(sent_n=200)


    

    ### save SL train/test data (general)
    # save_train_test_data(config, common_data)


    ### save ELISL benchmark data
    # prepare_ELISL_data(config, common_data, ELISL_data_dir="./data/benchmark/ELISL_data")

    ### obtain benchmark dataframes
    # get_benchmark_df(config, common_data)


    ### IDH1-DDR case study data
    # prepare_IDH1_DDR_data(common_data, cancer="LAML")


    ### save SL independent test data
        
    # data_total = prepare_SL_data(
    #     config=config,
    #     cancer="all",
    #     common_data=common_data,
    #     type="downstream"
    # )

    # save_independent_test_data(config, data_total)


    ### calculate random AUPR
    get_random_aupr_f1(config, common_data)

