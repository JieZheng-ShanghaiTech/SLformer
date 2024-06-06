import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import os
import yaml
import easydict
import argparse
import pickle as pkl
import networkx as nx
from tqdm import tqdm
import random
from datasets import load_from_disk
from datasets import Dataset

from util import create_dir
from dataset import GeneSentenceDataset



class Data_Preprocess():

    def __init__(
        self,
        config
    ):
        self.config = config

        self.construct_dirs()


    def construct_dirs(self):

        self.data_path_repository = {
            "sc_raw": os.path.join(self.config.sc_dir, "raw"),
            "sc_processed": os.path.join(self.config.sc_dir, "processed"),
            "map": os.path.join(self.config.SAVED_DATA_DIR, "map"),
            "emb": os.path.join(self.config.SAVED_DATA_DIR, "emb"),
            "coexp_data": os.path.join(self.config.sc_dir, "coexp_data"),
            "coexp_graph": os.path.join(self.config.sc_dir, "coexp_graph"),
            "genesent_root": os.path.join(self.config.sc_dir, "gene_sentence"),
        }

        for data_name, path in self.data_path_repository.items():
            if data_name != "sc_raw":
                create_dir(path)
        
        create_dir(os.path.join(self.data_path_repository["coexp_graph"], "graph"))
        create_dir(os.path.join(self.data_path_repository["coexp_graph"], "degree"))



    def check_data(self):
        # check whether all the data preprocessing is completed
        return
    

    def get_common_data(self, n):
        # extract gene emb map etc.
        # construct cancer2idmap, id2cancermap
        common_data_path = {
            "geneformer_emb_map": os.path.join(self.data_path_repository["map"], "geneformer_emb.pkl"),
            "geneformer_emb_mtx": os.path.join(self.data_path_repository["emb"], "geneformer_emb.npy"),
            "gene2sent_map": os.path.join(self.data_path_repository["map"], f"gene2sent_n{n}.pkl"),
            "sent_mask_map": os.path.join(self.data_path_repository["map"], f"sent_mask_n{n}.pkl"),
            "gene2id_map": os.path.join(self.data_path_repository["map"], "gene2id.pkl"),
            # "cancer_list": os.path.join(self.data_path_repository["map"], "cancer_list.txt"),
        }

        for data, path in common_data_path.items():
            if not os.path.exists(path):
                raise Exception(f"{data} cannot be found, please first construct it.")
        
        common_data = {}
        for data in ["geneformer_emb_map","gene2sent_map","sent_mask_map","gene2id_map"]:
            with open(common_data_path[data], 'rb') as f:
                common_data[data] = pkl.load(f)
        common_data["geneformer_emb_mtx"] = np.load(common_data_path["geneformer_emb_mtx"])
        
        cancer_list = list(self.config.sc_samples.keys())
        if "add_sc_samples" in self.config and len(self.config.add_sc_samples)>0:
            cancer_list += list(self.config.add_sc_samples.keys())
        cancer2id_map = {c:i for i,c in enumerate(cancer_list)}
        id2cancer_map = {i:c for i,c in enumerate(cancer_list)}

        common_data["cancer_list"] = cancer_list
        common_data["cancer2id_map"] = cancer2id_map
        common_data["id2cancer_map"] = id2cancer_map

        return common_data
    

    def data_prepare_sc(self, additional=False):

        if not additional:
            cancer_list = list(self.config.sc_samples.keys())

            # check if all the data has been processed
            all_data = True
            for cancer in cancer_list:
                fp = os.path.join(self.data_path_repository["sc_processed"],f"{cancer}_expression.h5ad")
                if not os.path.exists(fp):
                    all_data = False
                    break
            
            if not all_data:
                print("Start preprocessing sc data...")
                adata_total = ad.AnnData(np.zeros((0,0)))
                for cancer, sample_name in self.config.sc_samples.items():
                    adata = sc_preprocess(cancer, sample_name, self.data_path_repository["sc_raw"])
                    if len(adata_total) < 1:
                        adata_total = adata
                    else:
                        adata_total = adata_total.concatenate(adata)
                
                # prepare for geneformer preprocess
                geneformer_geneinfo = pd.read_csv(self.config.geneformer_gene_info_path, index_col=0)
                gene_ensembl_map = dict(zip(geneformer_geneinfo["gene_name"], geneformer_geneinfo["ensembl_id"]))
                with open(os.path.join(self.data_path_repository["map"], "gene2ensembl.pkl"), 'wb') as f:
                    pkl.dump(gene_ensembl_map, f)
                overlap_gene = list(set(adata_total.var_names).intersection(set(geneformer_geneinfo["gene_name"])))
                ensembl_ids = [gene_ensembl_map[g] for g in overlap_gene]
                for cancer in cancer_list:
                    adata_cancer = adata_total[adata_total.obs['cancer']==cancer]
                    adata_subset = adata_cancer[:, overlap_gene]
                    adata_subset.var["ensembl_id"] = ensembl_ids
                    adata_subset.obs["n_counts"] = np.array(adata_subset.X.astype(bool).sum(axis=1))
                    adata_subset.write_h5ad(os.path.join(self.data_path_repository["sc_processed"],f"{cancer}_expression.h5ad"))
                
                # gene/cancer list
                if not os.path.exists(os.path.join(self.data_path_repository["map"], "gene_list.txt")):
                    gene_list = overlap_gene
                    with open(os.path.join(self.data_path_repository["map"], "gene_list.txt"), 'w') as fp:
                        for g in list(gene_list):
                            fp.write("%s\n" % g)
                gene2id_map = {g:i for i, g in enumerate(gene_list)}
                with open(os.path.join(self.data_path_repository["map"], "gene2id.pkl"), 'wb') as f:
                    pkl.dump(gene2id_map, f)
                # cancer list
                with open(os.path.join(self.data_path_repository["map"], "cancer_list.txt"), 'w') as fp:
                    for c in list(cancer_list):
                        fp.write("%s\n" % c)
                    
        else:   # adding an additional cancer data
            cancer_list = list(self.config.add_sc_samples.keys())

            print("Start preprocessing additional sc data...")
            with open(os.path.join(self.data_path_repository["map"], "gene_list.txt")) as f:
                gene_list = [line.rstrip('\n') for line in f]
            with open(os.path.join(self.data_path_repository["map"], "gene2ensembl.pkl"), 'rb') as f:
                gene_ensembl_map = pkl.load(f)

            for cancer, sample_name in self.config.add_sc_samples.items():
                if not os.path.exists(os.path.join(self.data_path_repository["sc_processed"],f"{cancer}_expression.h5ad")):
                    adata = sc_preprocess(cancer, sample_name, self.data_path_repository["sc_raw"])
                    overlap_gene = list(set(adata.var_names).intersection(set(gene_list)))
                    adata_subset = adata[:, overlap_gene]
                    ensembl_ids = [gene_ensembl_map[g] for g in overlap_gene]
                    adata_subset.var["ensembl_id"] = ensembl_ids
                    adata_subset.obs["n_counts"] = np.array(adata_subset.X.astype(bool).sum(axis=1))
                    adata_subset.write_h5ad(os.path.join(self.data_path_repository["sc_processed"],f"{cancer}_expression.h5ad"))
            
        print("sc data processing is complete!")
    

    def data_prepare_coexp(self, additional=False):

        if not additional:
            cancer_list = list(self.config.sc_samples.keys())
        else:
            cancer_list = list(self.config.add_sc_samples.keys())

        # # coexp matrix
        # print("Start processing coexp data...")
        # for cancer in cancer_list:
        #     adata_cancer = sc.read_h5ad(os.path.join(self.data_path_repository["sc_processed"],f"{cancer}_expression.h5ad"))
        #     calc_coexp_corr(cancer, adata_cancer,
        #                     output_dir=self.data_path_repository["coexp_data"])
        
        # # coexp graph
        # print("Start processing coexp graphs...")
        # for cancer in cancer_list:
        #     construct_coexp_graph(cancer, 
        #                           coexp_dir=self.data_path_repository["coexp_data"], 
        #                           output_dir=self.data_path_repository["coexp_graph"], 
        #                           gene_list_file=os.path.join(self.data_path_repository["map"], "gene_list.txt"), 
        #                           percentile=99)

        # geneformer emb map
        if additional:
            cancer_list_origin = list(self.config.sc_samples.keys())
            cancer_list_full = cancer_list_origin+cancer_list
            cancer_list = cancer_list_full
        
        emb_loader = GeneformerEmb_Loader(
            emb_dir=self.config.geneformer_emb_dir,
            cancer_list=cancer_list,
            gene2ensembl_file=os.path.join(self.data_path_repository["map"], "gene2ensembl.pkl"),
            gene2id_file=os.path.join(self.data_path_repository["map"], "gene2id.pkl"),
        )

        gene_emb_map = emb_loader.integrate_emb()
        with open(os.path.join(self.data_path_repository["map"], "geneformer_emb.pkl"), 'wb') as f:
            pkl.dump(gene_emb_map, f)
        
        gene_emb_mtx = emb_loader.construct_emb_mtx(
            gene2emb_map_fp=os.path.join(self.data_path_repository["map"], "geneformer_emb.pkl"),
            add_padding=True
        )
        np.save(os.path.join(self.data_path_repository["emb"], "geneformer_emb.npy"), gene_emb_mtx)
    


    def data_prepare_genesent(self, n, additional=False):

        if not additional:
            cancer_list = list(self.config.sc_samples.keys())
        else:
            cancer_list = list(self.config.sc_samples.keys())+list(self.config.add_sc_samples.keys())

        # gene sentence
        dataset_genesent = construct_gene_sent(self.data_path_repository, cancer_list, n=n)
        gene_sent_map, sent_mask_map = dataset_genesent.get_gene_sent_map(return_mask=True)
        
        with open(os.path.join(self.data_path_repository["map"], f"gene2sent_n{n}.pkl"), 'wb') as f:
            pkl.dump(gene_sent_map, f)
        with open(os.path.join(self.data_path_repository["map"], f"sent_mask_n{n}.pkl"), 'wb') as f:
            pkl.dump(sent_mask_map, f)



def sc_preprocess(cancer, sample, sc_dir):

    f_h5 = sample+"_expression.h5"
    f_meta = sample+"_CellMetainfo_table.tsv"

    adata = sc.readwrite._read_v3_10x_h5(os.path.join(sc_dir, f_h5))
    meta_info = pd.read_csv(os.path.join(sc_dir, f_meta), sep='\t')

    # only use data of the malignant cells
    malignant_barcodes = meta_info[meta_info['Celltype (malignancy)']=='Malignant cells']['Cell']
    adata_malignant = adata[malignant_barcodes]
    adata_malignant.obs['cancer'] = cancer

    return adata_malignant


def spearman_corr(adata):
    data_df = adata.to_df()
    return data_df.corr(method='spearman')


def calc_coexp_corr(cancer, adata, output_dir):

    output_fp = os.path.join(output_dir, f"{cancer}_coexp.csv")
    if os.path.exists(output_fp):
        print(f"Found existing {cancer} coexp data!")
    else:
        print(f"***Computing coexp matrix of {cancer} data***")
        corr = spearman_corr(adata)
        corr.to_csv(os.path.join(output_dir, f"{cancer}_coexp.csv"))


def preprocess_coexp_df(coexp_df, thr, data="coexp_coefficient"):

    df = coexp_df.stack()
    df.index = df.index.rename('gene', level=1)
    df.name = data
    df = df.reset_index()
    df.columns = ['gene_a', 'gene_b', data]
    
    # drop gene_a==gene_b
    df = df[-(df['gene_a']==df['gene_b'])]
    # drop the duplicated lines
    df = df.drop_duplicates(subset=['gene_a', 'gene_b'], keep='first')
    
    # filter by coefficient threhold
    df = df[df["coexp_coefficient"]>thr]
    
    return df


def construct_coexp_graph(cancer_type, coexp_dir, output_dir, gene_list_file, percentile=99):

    fname = cancer_type
    graph_fp = os.path.join(output_dir, f"graph/{fname}_{percentile}_graph.csv")
    degree_fp = os.path.join(output_dir, f"degree/{fname}_{percentile}_graph.pkl")

    if os.path.exists(graph_fp):
        print(f"Found existing {cancer_type} coexp graph data!")
    
    else:

        coexp_df = pd.read_csv(os.path.join(coexp_dir, f"{fname}_coexp.csv"), index_col=0)

        thr = np.nanpercentile(coexp_df.values, percentile)
        df_filt = preprocess_coexp_df(coexp_df, thr=thr)

        print("***Building co-expression graphs of", cancer_type, "data with thr=", thr, "***")
        
        G = nx.from_pandas_edgelist(df_filt, 'gene_a', 'gene_b', ['coexp_coefficient'], create_using=nx.Graph())

        with open(gene_list_file) as f:
            gene_list = [line.rstrip('\n') for line in f]

        # add the genes which don't have any neighbors to the graph
        n_list = list(G.nodes())
        G.add_nodes_from(list(set(gene_list) - set(n_list)))

        # save edge list
        df_g = nx.to_pandas_edgelist(G)
        df_g.to_csv(graph_fp)

        # degrees
        degree_info = G.degree(gene_list)
        with open(degree_fp, 'wb') as f:
            pkl.dump(degree_info, f)


def construct_gene_sent(data_path_repository, cancer_list, n):

    n_genesent_dir = os.path.join(data_path_repository["genesent_root"], f"gene_sentence_n{n}")

    with open(os.path.join(data_path_repository["map"], "geneformer_emb.pkl"), 'rb') as f:
        geneformer_emb_map = pkl.load(f)

    gsentence_load = LoadGeneSentence(
        data_dir=os.path.join(data_path_repository["coexp_graph"], "graph"),
        output_dir=n_genesent_dir,
        cancer_list=cancer_list,
        gene2id_file=os.path.join(data_path_repository["map"], "gene2id.pkl"),
        geneformer_emb_map=geneformer_emb_map,
    )

    if not os.path.exists(n_genesent_dir):
        create_dir(n_genesent_dir)
        gsentence_load.process(
                max_nodes_sampling=n,
                thr=0.99,
                transform=True,
                filt_by_geneformer=True
            )

    dataset = gsentence_load.load()
    dataset_genesent = GeneSentenceDataset(dataset)

    return dataset_genesent





class LoadGeneSentence():

    def __init__(self, data_dir, output_dir, cancer_list, gene2id_file, geneformer_emb_map):
        
        self.data_dir = data_dir
        self.output_dir = output_dir

        with open(gene2id_file, 'rb') as f:
            self.gene2id_map = pkl.load(f)
        self.gene_list = list(self.gene2id_map.keys())
        self.gene_ids = list(self.gene2id_map.values())
        self.ngene = len(self.gene_list)

        self.geneformer_emb_map = geneformer_emb_map

        self.cancer_list = cancer_list

        self.cancer_id_map = {}
        for i, cancer in enumerate(self.cancer_list):
            self.cancer_id_map[cancer] = i

    
    def load(self):
        # load constructed dataset
        data = load_from_disk(self.output_dir)

        return data


    def process(self, cancer_input=None, max_nodes_sampling=200, thr=0.1, transform=False, filt_by_geneformer=True, random_order=False):

        # Start from beginning to construct subgraphs
        data_list = []

        if cancer_input is None:
            cancer_list = list(self.cancer_id_map.keys())
        else:
            cancer_list = [cancer_input]
        
        for cancer in cancer_list:
                
            prefix = cancer.replace("/", "_")
            fname = f"{prefix}_{thr}_graph.csv"

            cancer_idx = self.cancer_id_map[cancer]

            df = pd.read_csv(os.path.join(self.data_dir, fname))
            G = nx.from_pandas_edgelist(df, 'source', 'target', ['coexp_coefficient'], create_using=nx.Graph())

            # relabel nodes to int idx
            G_new = nx.relabel_nodes(G, self.gene2id_map, copy=True)

            for root_idx in tqdm(list(G_new.nodes()), desc="Process "+fname):
                if root_idx in self.gene_ids and root_idx in self.geneformer_emb_map[cancer_idx]:
                # if root_idx in self.gene_ids:
                
                    if random_order:
                        neighbors = list(G_new.neighbors(root_idx))
                        # random.shuffle(neighbors)
                        # sorted_neighbors = neighbors
                        sorted_neighbors = random.sample(self.gene_ids, len(neighbors))
                    else:
                        max_edges = sorted(G_new[root_idx].items(), key=lambda edge: edge[1]['coexp_coefficient'], reverse=True)
                        sorted_neighbors = [i[0] for i in max_edges]
                    if filt_by_geneformer:
                        sorted_neighbors_filt = [g for g in sorted_neighbors if g in self.geneformer_emb_map[cancer_idx]]
                        sorted_neighbors = sorted_neighbors_filt
                    if len(sorted_neighbors) > max_nodes_sampling:
                        # sampled_neighbors = list(random.sample(neighbors, max_nodes_sampling))
                        sampled_neighbors = sorted_neighbors[:max_nodes_sampling]
                    else:
                        sampled_neighbors = sorted_neighbors

                    # cancer label
                    cancer_idx = self.cancer_id_map[cancer]

                    # build gene sentences
                    if transform:
                        sentence_inst = padding_genesentence_transform([root_idx] + sampled_neighbors,
                                                                        cancer=cancer_idx, ngene=self.ngene, max_length=max_nodes_sampling+1)
                    else:
                        sentence_inst = padding_genesentence([root_idx] + sampled_neighbors, max_length=max_nodes_sampling+1)

                    meta_info = {
                        "cancer": cancer_idx,
                        "root_gene": root_idx
                    }

                    sentence_inst.update(meta_info)

                    data_list.append(sentence_inst)

                    # transfer nids: g_dgl.ndata['idx_transfer'] = g_dgl.ndata['idx']+cancer_idx*self.ngene
                        

        dataset = Dataset.from_list(data_list)
        dataset.save_to_disk(self.output_dir)

        print("Done!")



def padding_genesentence(gene_list, max_length, padding_id=0):

    # idx for each gene should be added by 1 since 0 is for padding id
    sentence = [g+1 for g in gene_list]
    if len(sentence) >= max_length:
        sentence = sentence[:max_length]
        att_mask = [1]*max_length
        length = max_length
    else:
        length = len(sentence)
        padding_seq = [padding_id]*(max_length-len(sentence))
        att_mask = [1]*len(sentence)+[0]*len(padding_seq)
        sentence += padding_seq

    sentence_inst = {"input_ids":sentence,
                     "attention_mask":att_mask,
                     "length": length}

    return sentence_inst


def padding_genesentence_transform(gene_list, cancer, ngene, max_length, padding_id=0):

    # idx for each gene is further transformed to adapt to the multi-cancer embedding
    try:
        sentence = [(g+1)+cancer*(ngene+1) for g in gene_list]
    except:
        print(gene_list, cancer, ngene)
    if len(sentence) >= max_length:
        sentence = sentence[:max_length]
        att_mask = [1]*max_length
        length = max_length
    else:
        length = len(sentence)
        padding_seq = [padding_id]*(max_length-len(sentence))
        att_mask = [1]*len(sentence)+[0]*len(padding_seq)
        sentence += padding_seq

    sentence_inst = {"input_ids":sentence,
                     "attention_mask":att_mask,
                     "length": length}

    return sentence_inst



class GeneformerEmb_Loader():

    def __init__(self, emb_dir, cancer_list, gene2ensembl_file, gene2id_file):

        self.dir = emb_dir

        self.cancer_list = cancer_list

        self.cancer_id_map = {}
        for i, cancer in enumerate(self.cancer_list):
            self.cancer_id_map[cancer] = i
        
        with open(gene2ensembl_file, 'rb') as f:
            self.gene2ensembl_map = pkl.load(f)

        with open(gene2id_file, 'rb') as f:
            self.gene2id_map = pkl.load(f)
        self.gene_list = list(self.gene2id_map.keys())

    
    def integrate_emb(self):

        gene_emb_map = {}
        # {cancer1: {g1:[], g2:[], ...}, cancer2:...}

        for cancer in self.cancer_list:
            fname = cancer
            emb_df = pd.read_csv(os.path.join(self.dir, fname+".csv"), index_col=0)
            cancer_idx = self.cancer_id_map[cancer]

            gene_emb_map[cancer_idx] = {}

            for g in self.gene_list:
                g_idx = self.gene2id_map[g]
                ensembl_id = self.gene2ensembl_map[g]
                if ensembl_id in emb_df.index:
                    emb = emb_df.loc[ensembl_id].values
                    if emb.ndim > 1:    # sometimes there are more than 1 embeddings for the same gene
                        emb = emb[0]
                    gene_emb_map[cancer_idx][g_idx] = emb
            
        return gene_emb_map
    

    # only use this when fill_norm is set to True
    def construct_emb_mtx(self, gene2emb_map_fp=None, add_padding=True):

        if os.path.exists(gene2emb_map_fp):
            with open(gene2emb_map_fp, 'rb') as f:
                gene2emb_map = pkl.load(f)
        else:
            gene2emb_map = self.integrate_emb()

        emb_data = []

        for cancer_idx in range(len(self.cancer_list)):
            if add_padding: # This is adapated to genesentence, g_idx in gene sentence starts from 1
                padding_emb = np.zeros(256)
                emb_data.append(padding_emb)
            for g, g_idx in self.gene2id_map.items():
                if g_idx in gene2emb_map[cancer_idx]:
                    emb = gene2emb_map[cancer_idx][g_idx]
                    emb_data.append(emb)
                else:
                    emb_data.append(np.zeros(256))

        return np.array(emb_data)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Data Preprocess')

    parser.add_argument('--config_file', type=str, default="./config/cancer_specific.yaml",
                        help='config file path')
    
    args = parser.parse_args()


    with open(args.config_file, 'r') as f:
        config = easydict.EasyDict(yaml.safe_load(f))

    data_preprocess = Data_Preprocess(config)

    # # preprocess single-cell data (TISCH2 data as an example)
    # data_preprocess.data_prepare_sc()

    # # go to geneformer_preprocess.ipynb to obtain geneformer embeddings
    # # need to indicate geneformer_emb_dir in the config file

    # # preprocess and prepare co-expression data
    # data_preprocess.data_prepare_coexp()

    # # preprocess and prepare gene sentence data
    data_preprocess.data_prepare_genesent(n=20)


    # ===== add an additional cancer type =====

    # data_preprocess.data_prepare_sc(additional=True)
    # data_preprocess.data_prepare_coexp(additional=True)
    # data_preprocess.data_prepare_genesent(n=10, additional=True)
    
    
    
