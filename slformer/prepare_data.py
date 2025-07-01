import argparse
import yaml
import easydict
import pandas as pd
import numpy as np
import os
import json
import random
from sklearn.metrics import average_precision_score, precision_recall_curve

from preprocess import Data_Preprocess
from dataloader import prepare_SL_data
from util import create_dir, get_train_test_SL, transfer_data_idx, find_data_idx


def save_train_test_data(config, common_data):

    cancer_list = common_data["cancer_list"]
    cancer_list = ["Glioma"]+cancer_list
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
        if cancer_type == "Glioma":
            print()
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
            

def save_mix_add_GBM_data(config, common_data):

    gene_sent_map = common_data["gene2sent_map"]
    geneformer_emb_map=common_data["geneformer_emb_map"]

    data_origin = prepare_SL_data(
            config=config,
            cancer="all",
            common_data=common_data,
            type="general"
        )

    # additionally add GBM data
    GBM_data = np.load("./data/saved_data/GBM_SL/GBM_SL.npy")
    filt_gbm_data = []
    for i in range(len(GBM_data)):
        g1_idx = GBM_data[i, 0]
        g2_idx = GBM_data[i, 1]
        if g1_idx in gene_sent_map[8] and g2_idx in gene_sent_map[8]:
            if g1_idx in geneformer_emb_map[8] and g2_idx in geneformer_emb_map[8]:
                filt_gbm_data.append([g1_idx, g2_idx, GBM_data[i, 2], 8])
    print("size of filt GBM data:", len(filt_gbm_data))
    filt_gbm_data = np.array(filt_gbm_data)

    save_dir = os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "mix_add_GBM_stratified")
    create_dir(save_dir)

    for cv in range(1,6):
        data_test_ori, data_train_ori = get_train_test_SL(data_origin, cv=cv, data_all=False, return_idx=False)
        data_test_glioma, data_train_glioma = get_train_test_SL(filt_gbm_data, cv=cv, data_all=False, return_idx=False)
        data_test = np.concatenate((data_test_ori, data_test_glioma), axis=0)
        data_train = np.concatenate((data_train_ori, data_train_glioma), axis=0)

        np.save(os.path.join(save_dir, f"test_all_fold_{cv}.npy"), data_test)
        np.save(os.path.join(save_dir, f"train_all_fold_{cv}.npy"), data_train)


def save_full_train_data(config, common_data):

    save_dir = os.path.join(config.SAVED_DATA_DIR, "SL_full_data")
    create_dir(save_dir)
    data_total = prepare_SL_data(
        config=config,
        cancer="all",
        common_data=common_data,
        type="general"
    )
    
    np.save(os.path.join(save_dir, "SL_full.npy"), data_total)


def save_fewshot_test_data(config, common_data):

    save_dir = os.path.join(config.SAVED_DATA_DIR, "SL_fewshot_data")
    create_dir(save_dir)

    cancer_list = common_data["cancer_list"]

    data_total_cancer = {}
    for cancer_type in cancer_list:
        data_total = prepare_SL_data(
            config=config,
            cancer=cancer_type,
            common_data=common_data,
            type="general"
        )
        data_total_cancer[cancer_type] = data_total

    # for cancer_out in cancer_list:
    for cancer_out in ['LAML','SKCM','CESC']:
    # cancer_out = 'OV'
    # cancer_out = 'BRCA'
        cancer_cv = [c for c in cancer_list if c != cancer_out]
        data_mix_train = {i:[] for i in range(1,6)}
        data_mix_test = {i:[] for i in range(1,6)}
        for cancer_type in cancer_cv:
            data_total = data_total_cancer[cancer_type]
            for cv in range(1,6):
                data_test, data_train = get_train_test_SL(data_total, cv=cv, data_all=False, return_idx=False)
                data_mix_train[cv].append(data_train)
                data_mix_test[cv].append(data_test)

        data_out = prepare_SL_data(
            config=config,
            cancer=cancer_out,
            common_data=common_data,
            type="general"
        )
        ## can change the ratio of test data here
        data_out_val, data_out_mix = get_train_test_SL(data_out, test_size=0.2, cv=1, data_all=False, return_idx=False)

        np.save(os.path.join(save_dir, f"fewshot_{cancer_out}_val.npy"), data_out_val)
        np.save(os.path.join(save_dir, f"fewshot_{cancer_out}_train.npy"), data_out_mix)

        for cv in range(1,6):
            ## add the part of the out cancer data to the training data
            data_out_test, data_out_train = get_train_test_SL(data_out_mix, test_size=0.2, cv=cv, data_all=False, return_idx=False)
            data_mix_train[cv].append(data_out_train)
            data_mix_test[cv].append(data_out_test)
            data_train = np.concatenate(data_mix_train[cv], axis=0)
            data_test = np.concatenate(data_mix_test[cv], axis=0)
            np.save(os.path.join(save_dir, f"test_{cancer_out}_fold_{cv}.npy"), data_test)
            np.save(os.path.join(save_dir, f"train_{cancer_out}_fold_{cv}.npy"), data_train)
    
    

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


## return gene names instead
def filt_SL_test_names(gene_candidates, gene2id_map, gene_emb_map, geneformer_emb_map, context=8):

    id2gene_map = {i:g for g,i in gene2id_map.items()}

    candidate_filt = list(set(gene_candidates).intersection(set(gene2id_map.keys())))
    candidate_id = [gene2id_map[g] for g in candidate_filt]

    gene_context = list(gene_emb_map[context].keys())
    gene_context_geneformer = list(geneformer_emb_map[context].keys())
    gene_context_overlap = list(set(gene_context).intersection(set(gene_context_geneformer)))
    candidate_id_filt = list(set(candidate_id).intersection(set(gene_context_overlap)))
    candidate_names_filt = [id2gene_map[i] for i in candidate_id_filt]

    return candidate_names_filt


def filt_SL_test_pairs(gene1_candidates, gene2_candidates, gene2id_map, gene_emb_map, geneformer_emb_map, context=8):

    candidate_filt = [(a, b) for a, b in zip(gene1_candidates, gene2_candidates) if a in gene2id_map.keys() and b in gene2id_map.keys()]
    candidate_id = [(gene2id_map[g1], gene2id_map[g2]) for (g1,g2) in candidate_filt]

    gene_context = list(gene_emb_map[context].keys())
    gene_context_geneformer = list(geneformer_emb_map[context].keys())
    gene_context_overlap = list(set(gene_context).intersection(set(gene_context_geneformer)))

    candidate_id_filt = [(a, b) for a, b in candidate_id if a in gene_context_overlap and b in gene_context_overlap]

    return candidate_id_filt


def filt_SL_test_pairs_id(gene1_candidates, gene2_candidates, gene2id_map, gene_emb_map, geneformer_emb_map, context=8):

    candidate_id = [(a, b) for a, b in zip(gene1_candidates, gene2_candidates)]
    gene_context = list(gene_emb_map[context].keys())
    gene_context_geneformer = list(geneformer_emb_map[context].keys())
    gene_context_overlap = list(set(gene_context).intersection(set(gene_context_geneformer)))

    candidate_id_filt = [(a, b) for a, b in candidate_id if a in gene_context_overlap and b in gene_context_overlap]

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
    ## add additional candidates genes from https://link.springer.com/article/10.1007/s11912-020-01006-6
    data_reactome.append([gene2id_map['IDH1'], gene2id_map['PDGFRA'], 0, cancer_id])
    data_reactome.append([gene2id_map['IDH1'], gene2id_map['RB1'], 0, cancer_id])
    data_reactome.append([gene2id_map['IDH1'], gene2id_map['PIK3CA'], 0, cancer_id])

    data_kegg = []
    for g in gene_kegg_filt:
        data_kegg.append([gene2id_map['IDH1'], g, 0, cancer_id])

    np.save(os.path.join(save_dir, f"IDH1_DDR_{cancer}_reactome.npy"), np.array(data_reactome))
    np.save(os.path.join(save_dir, f"IDH1_DDR_{cancer}_kegg.npy"), np.array(data_kegg))


def prepare_PTEN_GBM_data(common_data, cancer="Glioma"):

    save_dir = "./data/saved_data/inference"
    create_dir(save_dir)

    gene_sent_map = common_data["gene2sent_map"]
    geneformer_emb_map=common_data["geneformer_emb_map"]
    gene2id_map = common_data["gene2id_map"]
    cancer2id_map = common_data["cancer2id_map"]

    genes_all = list(gene2id_map.keys())
    cancer_id = cancer2id_map[cancer]
    genes_all_filt = filt_SL_test(genes_all, gene2id_map, gene_sent_map, geneformer_emb_map, context=cancer_id)

    print("total number of partner genes:", len(genes_all_filt))
    ## 7301
    data_all = []
    for g in genes_all_filt:
        data_all.append([gene2id_map['PTEN'], g, 0, cancer_id])

    np.save(os.path.join(save_dir, f"PTEN_{cancer}_allgenes.npy"), np.array(data_all))



def IDH1_permute_data(common_data, partner_candidate, n_sample=100, seed=0, cancer="Glioma"):

    gene_sent_map = common_data["gene2sent_map"]
    geneformer_emb_map=common_data["geneformer_emb_map"]
    gene2id_map = common_data["gene2id_map"]
    cancer2id_map = common_data["cancer2id_map"]

    genes_all = list(gene2id_map.keys())

    ## sample the background genes
    random.seed(seed)
    genes_bkg = random.sample(genes_all, n_sample)
    # test_genes = [partner_candidate]+genes_bkg

    cancer_id = cancer2id_map[cancer]
    # test_genes_filt = filt_SL_test(test_genes, gene2id_map, gene_sent_map, geneformer_emb_map, context=cancer_id)
    test_genes_filt = filt_SL_test(genes_bkg, gene2id_map, gene_sent_map, geneformer_emb_map, context=cancer_id)
    test_genes_filt = [gene2id_map[partner_candidate]]+test_genes_filt
    data_all = []
    for g in test_genes_filt:
        data_all.append([gene2id_map['IDH1'], g, 0, cancer_id])
    
    return np.array(data_all)



def prepare_GBM_data(common_data):

    data = pd.read_csv("./data/saved_data/GBM_SL/GBM_SLdata.csv")
    save_dir = "./data/saved_data/GBM_SL"

    gene_sent_map = common_data["gene2sent_map"]
    geneformer_emb_map=common_data["geneformer_emb_map"]
    gene2id_map = common_data["gene2id_map"]
    cancer2id_map = common_data["cancer2id_map"]
    id2gene_map = {i:g for g,i in gene2id_map.items()}

    # gene_emb_map=common_data["gene2sent_map"],
    # geneformer_emb_map=common_data["geneformer_emb_map"],

    cancer_id = cancer2id_map["Glioma"]

    filt_gene_pairs = filt_SL_test_pairs(list(data['Gene A']), list(data['Gene B']), gene2id_map, gene_sent_map, geneformer_emb_map, context=cancer_id)

    data_total = []
    print("# pos samples:", len(filt_gene_pairs))
    ## positive samples
    for g1, g2 in filt_gene_pairs:
        data_total.append([g1, g2, 1, cancer_id])
        # print(id2gene_map[g1], id2gene_map[g2])
    ## random negative samples
    random.seed(1)
    gene_context = list(geneformer_emb_map[cancer_id])
    gene1_rand = random.sample(gene_context, 50)
    gene2_rand = random.sample(gene_context, 50)

    neg_filt_gene_pairs = filt_SL_test_pairs_id(gene1_rand, gene2_rand, gene2id_map, gene_sent_map, geneformer_emb_map, context=cancer_id)

    for g1, g2 in neg_filt_gene_pairs:
        data_total.append([g1, g2, 0, cancer_id])

    print("# total samples:", len(data_total))
    data_total = np.array(data_total)
    np.save(os.path.join(save_dir, f"GBM_SL.npy"), data_total)

    ## filt through geneformer emb map and gene sent map
    # filt_data = []
    # for i in range(len(data_total)):
    #     g1_idx = data_total[i, 0]
    #     g2_idx = data_total[i, 1]
    #     if g1_idx in gene_sent_map[cancer_id] and g2_idx in gene_sent_map[cancer_id]:
    #         if g1_idx in geneformer_emb_map[cancer_id] and g2_idx in geneformer_emb_map[cancer_id]:
    #             filt_data.append([g1_idx, g2_idx, data_total[i, 2], 8])
    # print("size of filt GBM data:", len(filt_data))

    # return data_total

    # id2gene_map = {i:g for g,i in gene2id_map.items()}
    # GBM_data = np.load("./data/saved_data/GBM_SL/GBM_SL.npy")
    # for i in range(len(GBM_data)):
    #     print(id2gene_map[GBM_data[i,0]], id2gene_map[GBM_data[i,1]], GBM_data[i,2])


def prepare_MiSL_IDH1_data(cancer='LAML'):

    data = pd.read_excel("data/saved_data/MiSL_IDH1/MiSL_IDH1.xlsx")
    save_dir = "./data/saved_data/MiSL_IDH1/"

    gene_sent_map = common_data["gene2sent_map"]
    geneformer_emb_map=common_data["geneformer_emb_map"]
    gene2id_map = common_data["gene2id_map"]
    cancer2id_map = common_data["cancer2id_map"]
    id2gene_map = {i:g for g,i in gene2id_map.items()}

    misl_candidates = list(data['MiSL.Candidate'])
    cancer_id = cancer2id_map[cancer]
    misl_filt = filt_SL_test(misl_candidates, gene2id_map, gene_sent_map, geneformer_emb_map, context=cancer_id)
    print("# pos samples:", len(misl_filt))

    data_total = []
    for g in misl_filt:
        data_total.append([gene2id_map['IDH1'], g, 1, cancer_id])

    random.seed(1)
    gene_context = list(geneformer_emb_map[cancer_id])
    gene_context_sample = [g for g in gene_context if g not in misl_filt]
    neg_gene_ids = random.sample(gene_context_sample , 5*len(misl_filt))
    neg_gene = [id2gene_map[i] for i in neg_gene_ids]
    neg_filt = filt_SL_test(neg_gene, gene2id_map, gene_sent_map, geneformer_emb_map, context=cancer_id)
    neg_filt = neg_filt[:4*len(misl_filt)]  ## keep only 5* size in total
    for g in neg_filt:
        data_total.append([gene2id_map['IDH1'], g, 0, cancer_id])

    print("# total samples:", len(data_total))
    data_total = np.array(data_total)
    np.save(os.path.join(save_dir, f"MiSL_IDH1_processed_{cancer}.npy"), data_total)



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

    ### save mix SL train/test data (general+GBM)
    # save_mix_add_GBM_data(config, common_data)

    ## save full SL data (general)
    # save_full_train_data(config, common_data)

    ## save SL data for few shot test
    # save_fewshot_test_data(config, common_data)

    ### save ELISL benchmark data
    # prepare_ELISL_data(config, common_data, ELISL_data_dir="./data/benchmark/ELISL_data")

    ### obtain benchmark dataframes
    # get_benchmark_df(config, common_data)


    ### IDH1-DDR case study data
    # prepare_IDH1_DDR_data(common_data, cancer="COAD")

    ### GBM cancer SL data
    # prepare_GBM_data(common_data)

    ## MiSL candidates of IDH1
    # prepare_MiSL_IDH1_data(cancer='LAML')

    ## PTEN-SL in GBM
    prepare_PTEN_GBM_data(common_data, cancer="Glioma")

    ### save SL independent test data
        
    # data_total = prepare_SL_data(
    #     config=config,
    #     cancer="all",
    #     common_data=common_data,
    #     type="downstream"
    # )

    # save_independent_test_data(config, data_total)


    ### calculate random AUPR
    # get_random_aupr_f1(config, common_data)

