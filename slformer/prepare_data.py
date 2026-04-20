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
from util import create_dir, get_train_test_SL


class SL_Loader():

    def __init__(self, config, gene2id_map, gene_emb_map, geneformer_emb_map, cancer2id_map, type="general"):
        """
        Organizing SL gene pairs data
        args:
            - config: config file, should include paths and other information of SL datasets that need to be processed
            - gene2id_map: gene to index mapping
            - gene_emb_map: gene to gene sentence mapping
            - geneformer_emb_map: gene to Geneformer embedding mapping
            - cancer2id_map: cancer to index mapping
            - type: 'general' (benchmark data) or 'downstream' (independent test data)
        """

        self.SL_datasets = config.SL_dataset

        self.gene2id_map = gene2id_map
        self.gene_list = list(self.gene2id_map.keys())

        self.gene_emb_map = gene_emb_map
        self.geneformer_emb_map = geneformer_emb_map

        self.cancer2id_map = cancer2id_map

        # load ELISL SL dataset
        self.SL_general_df = pd.read_excel(self.SL_datasets.general.path)
        
        if type == "downstream":
            SL_general_data = self.get_SL_data(data_type="general",cancer_filt="all")
            self.SL_unique_gene = np.unique(SL_general_data[:,:2].flatten()).tolist()
            self.SL_general_map = self.construct_SL_general_map(self.SL_general_df)


    def get_SL_data(self, data_type, cancer_filt='all', downstream_stat=False):
        """
        returns:
            npy format of SL gene pairs data:
                [gene1_id, gene2_id, SL_label, context_id]
        """

        if data_type == "general":
            SL_general_data = self.construct_data(self.SL_general_df, cancer_filt)
            
            return SL_general_data
        
        elif data_type in self.SL_datasets.keys():
            SL_data = pd.read_csv(self.SL_datasets[data_type].path)
            SL_filt_general = self.filt_SL_general(self.SL_general_map, SL_data)

            SL_filt = self.construct_data(SL_filt_general, cancer_filt='all')
            print("Processing", data_type, "size=", len(SL_filt))

            if downstream_stat:
                downstream_gene = list(set(SL_filt[:,0]).union(set(SL_filt[:,1])))
                downstream_overlap = list(set(downstream_gene).intersection(set(self.SL_unique_gene)))
                print(f"Overlapped genes with ELISL SL data: {len(downstream_overlap)}/{len(downstream_gene)}")

            return SL_filt

        else:
            raise Exception("Invalid data type")
                

    def construct_data(self, df, cancer_filt):

        label_name = df.columns[-1]

        # leave out genes not included in gene list
        gene1_bool = [True if g in self.gene_list else False for g in df["gene1"]]
        gene2_bool = [True if g in self.gene_list else False for g in df["gene2"]]
        SL_filt = df[np.logical_and(gene1_bool, gene2_bool)]

        # leave out cancers not included in the cancer types
        if cancer_filt != 'all':
            cancer_bool = [True if cancer in cancer_filt else False for cancer in SL_filt["cancer"]]
            SL_filt = SL_filt[cancer_bool]

        # leave out genes without geneformer embeddings or specific embeddings
        filt_data = []
        for idx, row in SL_filt.iterrows():
            cancer = self.cancer2id_map[row["cancer"]]
            g1_idx = self.gene2id_map[row["gene1"]]
            g2_idx = self.gene2id_map[row["gene2"]]

            if g1_idx in self.gene_emb_map[cancer] and g2_idx in self.gene_emb_map[cancer]:
                if g1_idx in self.geneformer_emb_map[cancer] and g2_idx in self.geneformer_emb_map[cancer]:
                    filt_data.append([g1_idx, g2_idx, row[label_name], cancer])

        return np.array(filt_data)
    

    def construct_SL_general_map(self, SL_general_data):

        SL_map = {}
        for g in list(set(SL_general_data["gene1"]).union(set(SL_general_data["gene2"]))):
            SL_map[g] = []

        for idx, row in SL_general_data.iterrows():
            if row["gene2"] not in SL_map[row["gene1"]]:
                SL_map[row["gene1"]].append(row["gene2"])
            if row["gene1"] not in SL_map[row["gene2"]]:
                SL_map[row["gene2"]].append(row["gene1"])
        
        return SL_map
    

    def filt_SL_general(self, SL_general_map, SL_data):

        flag = []

        for idx, row in SL_data.iterrows():
            g1, g2 = row["gene1"], row["gene2"]
            if g1 in list(SL_general_map.keys()):
                if g2 not in SL_general_map[g1]:
                    flag.append(True)
                else:
                    flag.append(False)
            else:
                flag.append(True)
        
        return SL_data[flag]


def prepare_SL_data(config, cancer, common_data, type="general"):
    """
    function for getting npy SL pair data
    args:
        - config: config information for data processing
        - cancer: a specific cancer type or 'all'
        - common data: commonly used data returned by Data_Preprocess.get_common_data
        - type: 'general' (benchmark data) or 'downstream' (independent test data)
    """

    SL_loader = SL_Loader(
        config=config,
        gene2id_map=common_data["gene2id_map"],
        gene_emb_map=common_data["gene2sent_map"],
        geneformer_emb_map=common_data["geneformer_emb_map"],
        cancer2id_map=common_data["cancer2id_map"],
        type=type
    )

    ## benchmark data
    if type == "general":
        data_total = SL_loader.get_SL_data(data_type="general", cancer_filt=cancer) # return a numpy array
        if len(data_total) > 0: # belonging to ELISL cancer types
            print(f"Processed {cancer} data, size={len(data_total)}")
    ## independent test data
    elif type == "downstream":
        data_total = {}
        for data_type in list(SL_loader.SL_datasets.keys()):
            if data_type != "general":
                data = SL_loader.get_SL_data(data_type=data_type, cancer_filt="all", downstream_stat=False)
                data_total[data_type] = data

    return data_total


def save_train_test_data(config, common_data, suffix='all'):
    """
    save SL train/test data (8 cancers not including GBM)
    args:
        - config: config information for data processing
        - common data: commonly used data returned by Data_Preprocess.get_common_data
    """

    cancer_list = common_data["cancer_list"]
    id2cancer_map = {i:c for c,i in common_data["cancer2id_map"].items()}

    ## cancer-specific and mix
    save_dir = os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", f"cancer_specific_{suffix}")
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
    save_dir = os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", f"mix_{suffix}")
    create_dir(save_dir)
    for cv in range(1,6):
        data_train = np.concatenate(data_mix_train[cv], axis=0)
        data_test = np.concatenate(data_mix_test[cv], axis=0)
        np.save(os.path.join(save_dir, f"test_all_fold_{cv}.npy"), data_test)
        np.save(os.path.join(save_dir, f"train_all_fold_{cv}.npy"), data_train)
            
    ## cross-cancer
    save_dir = os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", f"cross_cancer_{suffix}")
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
    """
    save mixed-cancer scenario SL train/test data (9 cancers in total including GBM)
    """

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


def save_mix_subset_data(config, common_data, cancers):
    """
    save mixed-cancer scenario SL train/test data for only few cancer types
    Note 'cancers' should not include Glioma
    """
    cancer_types = cancers
    cancer_types.sort()
    save_dir = os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", f"mix_{'_'.join(cancer_types)}")
    create_dir(save_dir)
    
    data = prepare_SL_data(
        config=config,
        cancer=cancer_types,
        common_data=common_data,
        type="general"
    )
    print('data size', len(data))
    for cv in range(1,6):
        data_test, data_train = get_train_test_SL(data, test_size=0.2, cv=cv, data_all=False, return_idx=False)
        np.save(os.path.join(save_dir, f"test_all_fold_{cv}.npy"), data_test)
        np.save(os.path.join(save_dir, f"train_all_fold_{cv}.npy"), data_train)


def save_full_train_data(config, common_data):
    """
    save full SL data (8 cancers)
    """

    save_dir = os.path.join(config.SAVED_DATA_DIR, "SL_full_data")
    create_dir(save_dir)
    ## all cancers included in benchmark dataset
    data_total = prepare_SL_data(
        config=config,
        cancer="all",
        common_data=common_data,
        type="general"
    )
    
    np.save(os.path.join(save_dir, "SL_full.npy"), data_total)
    

def save_independent_test_data(config, common_data):
    """
    save SL independent test data
    """

    data_total = prepare_SL_data(
        config=config,
        cancer="all",
        common_data=common_data,
        type="downstream"
    )

    save_dir = os.path.join(config.SAVED_DATA_DIR, "independent_test_data")
    create_dir(save_dir)

    for data_type, data in data_total.items():
        np.save(os.path.join(save_dir, f"{data_type}.npy"), data)


def transfer_data_idx(data, gene2id_map, id2cancer_map, label_name='label'):
    """
    helper function for remapping gene indices
    """

    id2gene_map = {i:g for g,i in gene2id_map.items()}

    gene1_data, gene2_data, label_data, cancer_data = data[:,0],data[:,1],data[:,2],data[:,3]
    gene1_transfer = np.array([id2gene_map[gene1_data[i]] for i in range(len(gene1_data))]).reshape(-1,1)
    gene2_transfer = np.array([id2gene_map[gene2_data[i]] for i in range(len(gene2_data))]).reshape(-1,1)
    cancer_transfer = np.array([id2cancer_map[cancer_data[i]] for i in range(len(cancer_data))]).reshape(-1,1)

    data_transfer = np.concatenate([gene1_transfer, gene2_transfer, label_data.reshape(-1,1), cancer_transfer], axis=1)
    df = pd.DataFrame(data_transfer, columns=['gene1', 'gene2', label_name, 'cancer'])

    return df


def find_data_idx(subset_data, total_data):
    """
    helper function for indexing data
    """

    idx = []
    for i in range(len(subset_data)):
        idx.append(int(np.where((total_data == subset_data[i]).all(axis=1))[0][0]))

    return idx


def get_benchmark_df(config, common_data):
    """
    Organize unified benchmark SL pairs into dataframes
    """

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
        data_test = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cross_cancer_all", f"test_{cancer_name}.npy"))
        data_train = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cross_cancer_all", f"train_{cancer_name}.npy"))
        df_test = transfer_data_idx(data_test, gene2id_map, id2cancer_map)
        df_train = transfer_data_idx(data_train, gene2id_map, id2cancer_map)
        df_test.to_csv(os.path.join(save_dir, "cross_cancer_all", f"test_{cancer_name}.csv"), index=False)
        df_train.to_csv(os.path.join(save_dir, "cross_cancer_all", f"train_{cancer_name}.csv"), index=False)


def prepare_ELISL_data(config, common_data, ELISL_data_dir):
    """
    Organize benchmark SL pairs into dataframes that fit ELISL SL data format 
    """

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

        fold_idx = {}

        data_test = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cross_cancer", f"test_{cancer_name}.npy"))
        data_train = np.load(os.path.join(config.SAVED_DATA_DIR, "SL_train_test_data", "cross_cancer", f"train_{cancer_name}.npy"))
        test_idx = find_data_idx(data_test, data_all)
        train_idx = find_data_idx(data_train, data_all)

        fold_idx[str(0)] = {"train":list(train_idx), "test":list(test_idx)}
    
        with open(os.path.join(idx_dir, f"{cancer_name}_transfer_idx.json"), 'w') as fp:
            json.dump(fold_idx, fp)


"""
some helper functions for filtering out genes with missing features
"""
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
    """
    construct npy input data for IDH1-DDR genes SL pairs
    """

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
    ## add additional candidates genes from review: https://link.springer.com/article/10.1007/s11912-020-01006-6
    data_reactome.append([gene2id_map['IDH1'], gene2id_map['PDGFRA'], 0, cancer_id])
    data_reactome.append([gene2id_map['IDH1'], gene2id_map['RB1'], 0, cancer_id])
    data_reactome.append([gene2id_map['IDH1'], gene2id_map['PIK3CA'], 0, cancer_id])

    data_kegg = []
    for g in gene_kegg_filt:
        data_kegg.append([gene2id_map['IDH1'], g, 0, cancer_id])

    np.save(os.path.join(save_dir, f"IDH1_DDR_{cancer}_reactome.npy"), np.array(data_reactome))
    np.save(os.path.join(save_dir, f"IDH1_DDR_{cancer}_kegg.npy"), np.array(data_kegg))


def prepare_IDH1_PRKDC_data(common_data):
    """
    construct npy input data for IDH1-PRKDC SL pair
    """

    save_dir = "./data/saved_data/inference"
    create_dir(save_dir)

    gene_sent_map = common_data["gene2sent_map"]
    geneformer_emb_map=common_data["geneformer_emb_map"]
    gene2id_map = common_data["gene2id_map"]

    geneid1 = gene2id_map["IDH1"]; geneid2 = gene2id_map["PRKDC"]

    data_all = []
    for cancer_id in range(9):
        ## check if both genes are accessible
        genes_filt = filt_SL_test(["IDH1", "PRKDC"], gene2id_map, gene_sent_map, geneformer_emb_map, context=cancer_id)
        if len(genes_filt) >= 2:
            print(cancer_id)    ## 0,1,2,5,6,7,8
            data_all.append([geneid1, geneid2, 0, cancer_id]) ## dummpy label

    np.save(os.path.join(save_dir, f"IDH1_PRKDC_allcancer.npy"), np.array(data_all))


def single_permute_data(common_data, primary_gene, partner_candidate, n_sample=100, seed=0, cancer="Glioma"):
    """
    single-step of bootstrap ranking
    """

    gene_sent_map = common_data["gene2sent_map"]
    geneformer_emb_map=common_data["geneformer_emb_map"]
    gene2id_map = common_data["gene2id_map"]
    cancer2id_map = common_data["cancer2id_map"]

    genes_all = list(gene2id_map.keys())

    ## sample the background genes
    random.seed(seed)
    genes_bkg = random.sample(genes_all, n_sample)

    cancer_id = cancer2id_map[cancer]
    test_genes_filt = filt_SL_test(genes_bkg, gene2id_map, gene_sent_map, geneformer_emb_map, context=cancer_id)
    test_genes_filt = [gene2id_map[partner_candidate]]+test_genes_filt
    data_all = []
    for g in test_genes_filt:
        data_all.append([gene2id_map[primary_gene], g, 0, cancer_id])
    
    return np.array(data_all)



def prepare_GBM_data(common_data):
    """
    prepare additional GBM SL pairs data
    """

    data = pd.read_csv("./data/saved_data/GBM_SL/GBM_SLdata.csv")
    save_dir = "./data/saved_data/GBM_SL"

    gene_sent_map = common_data["gene2sent_map"]
    geneformer_emb_map=common_data["geneformer_emb_map"]
    gene2id_map = common_data["gene2id_map"]
    cancer2id_map = common_data["cancer2id_map"]
    id2gene_map = {i:g for g,i in gene2id_map.items()}

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


def get_random_aupr_f1(config, common_data):
    """
    compute performance metrics for random prediction
    """

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



def main(config):

    ## only run codes that are used
    data_preprocess = Data_Preprocess(config)
    common_data = data_preprocess.get_common_data(sent_n=200)

    ## save SL train/test data (general)
    save_train_test_data(config, common_data)

    ## save independent test
    save_independent_test_data(config, common_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='prepare SL data')
    parser.add_argument('--data_config_file', type=str, default="./config/data_preprocess.yaml",
                    help='data preprocess config file path')
    args = parser.parse_args()

    with open(args.data_config_file, 'r') as f:
        config = easydict.EasyDict(yaml.safe_load(f))

    main(config)
    

