import os
import pandas as pd
import numpy as np
import pickle as pkl
from scipy import stats
from torch.utils.data import Dataset, DataLoader

from dataset import SL_Dataset, GeneSentenceDataset
from util import split_data, split_data_by_cancer, get_weighted_sampler


def prepare_SL_data(config, cancer, common_data, type="general"):

    SL_loader = SL_Loader(
        config=config,
        gene2id_map=common_data["gene2id_map"],
        gene_emb_map=common_data["gene2sent_map"],
        geneformer_emb_map=common_data["geneformer_emb_map"],
        cancer2id_map=common_data["cancer2id_map"],
        type=type
    )

    if type == "general":
        data_total = SL_loader.get_SL_data(data_type="general", cancer_filt=cancer) # return a numpy array
        if len(data_total) > 0: # belonging to ELISL cancer types
            print(f"Processed {cancer} data, size={len(data_total)}")
    elif type == "downstream":
        data_total = {}
        for data_type in list(SL_loader.SL_datasets.keys()):
            if data_type != "general":
                data = SL_loader.get_SL_data(data_type=data_type, cancer_filt="all", downstream_stat=True)
                data_total[data_type] = data

    return data_total


def load_all_data_SL(all_data, gene_rpr_map, batch_size, n=None, anchor=True, bi_rpr=False, sent_mask=None, emb_mtx=None, augmentation=None):

    loader = DataLoader(SL_Dataset(all_data, gene_rpr_map, n=n, anchor=anchor, bi_rpr=bi_rpr, sent_mask=sent_mask, emb_mtx=emb_mtx, augmentation=augmentation), batch_size=batch_size, shuffle=False)
    
    return loader


def load_train_data_SL(test_data, train_data, gene_rpr_map, batch_size, n=None, anchor=True, bi_rpr=False, sent_mask=None, emb_mtx=None, augmentation=None):

    # sampler_test = get_weighted_sampler(test_data)
    # sampler_train = get_weighted_sampler(train_data)

    train_dataset = SL_Dataset(train_data, gene_rpr_map, n=n, anchor=anchor, bi_rpr=bi_rpr, sent_mask=sent_mask, emb_mtx=emb_mtx, augmentation=augmentation)
    test_dataset = SL_Dataset(test_data, gene_rpr_map, n=n, anchor=anchor, bi_rpr=bi_rpr, sent_mask=sent_mask, emb_mtx=emb_mtx, augmentation=None)

    drop_last = {"train":False, "test":False}
    for type, dataset in {"train":train_dataset, "test":test_dataset}.items():
        if len(dataset)%batch_size < 20:   # avoid the case that the last batch is too small
            drop_last[type] = True

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train, drop_last=drop_last["train"])
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler_test, drop_last=drop_last["test"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last["train"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last["test"])

    return train_loader, test_loader


def load_pretrain_data(train_data, test_data, batch_size, emb_mtx, n, gene2anno_map, random_init):

    train_dataset = GeneSentenceDataset(train_data, emb_mtx=emb_mtx, n=n, gene2anno_map=gene2anno_map, random_init=random_init)
    test_dataset = GeneSentenceDataset(test_data, emb_mtx=emb_mtx, n=n, gene2anno_map=gene2anno_map, random_init=random_init)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_pretrain_data_all(all_data, batch_size, emb_mtx, n, gene2anno_map, random_init):

    all_dataset = GeneSentenceDataset(all_data, emb_mtx=emb_mtx, n=n, gene2anno_map=gene2anno_map, random_init=random_init)

    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)

    return all_loader



class SL_Loader():

    # type = "general" or "downstream"
    def __init__(self, config, gene2id_map, gene_emb_map, geneformer_emb_map, cancer2id_map, type="general"):

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
            # SL_unique_gene = list(set(self.SL_general_df["gene1"]).union(set(self.SL_general_df["gene2"])))
            # SL_unique_gene = list(set(SL_unique_gene).intersection(set(self.gene_list)))
            # self.SL_unique_gene = [self.gene2id_map[g] for g in SL_unique_gene]


    def get_SL_data(self, data_type, cancer_filt='all', downstream_stat=False):

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