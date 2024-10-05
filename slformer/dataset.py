import torch
import numpy as np
import random

from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle

with open('/home/tinglu/LLM4SL/KG/slformer-gene2kgemb/geneid2kgemb256_p1.pkl', 'rb') as f:
    geneid2kgemb = pickle.load(f)
kg_emb_size = 256 # 128, 768
kg_sentence = 1

def get_swap_idx(n, swap_times=1):

    swap_idx = []

    for i in range(swap_times):
        # l_idx = random.sample(range(1, n-2), sample)
        if n <= 2:
            return [(0,0)]
        elif n==3:
            return [(1,2)]
        else:
            l_idx = random.choice(range(1, n-2))
            r_idx = l_idx+1

        swap_idx.append((l_idx, r_idx))

    return swap_idx


def get_mask_idx(n, mask_times=1):

    mask_idx = []

    for i in range(mask_times):

        idx = random.choice(range(1, n-1))
        mask_idx.append(idx)

    return mask_idx


class SL_Dataset(Dataset):

    def __init__(self, data, gene_rpr_map, n=None, sent_mask=None, emb_mtx=None, bi_rpr=False, augmentation=None, augment_fold=5):

        if augmentation is not None and n is not None:
            data = np.repeat(data, repeats=augment_fold, axis=0)

        self.augmentation = augmentation
        self.augment_fold = augment_fold
        self.data = data
        self.gene_rpr_map = gene_rpr_map
        self.n = n
        if n is not None and n > 200:
            raise Exception("Please set a valid gene sentence length (<=200)")

        self.sent_mask = sent_mask
        if emb_mtx is not None:
            self.emb_mtx = torch.tensor(emb_mtx)
        self.bi_rpr = bi_rpr

    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, idx):

        data_single = self.data[idx]
        # [g1, g2, label, cancer]
        cancer = data_single[3]

        if self.n is not None:  # use gene sentence
            rpr1 = self.gene_rpr_map[cancer][data_single[0]][:self.n]
            rpr2 = self.gene_rpr_map[cancer][data_single[1]][:self.n]
            if self.augmentation is not None and random.uniform(0,1) > 1/self.augment_fold: # otherwise remain unchanged
                if self.augmentation == "swap":
                    sent_len1 = self.sent_mask[cancer][data_single[0]][:self.n].count(1)
                    sent_len2 = self.sent_mask[cancer][data_single[1]][:self.n].count(1)
                    swap_idx1 = get_swap_idx(sent_len1, swap_times=5)
                    swap_idx2 = get_swap_idx(sent_len2, swap_times=5)
                    for l,r in swap_idx1:
                        rpr1[l], rpr1[r] = rpr1[r], rpr1[l]
                    for l,r in swap_idx2:
                        rpr2[l], rpr2[r] = rpr2[r], rpr2[l]
            # print("gene sentence1", rpr1)
            # print("gene sentence2", rpr2)
            rpr1 = torch.tensor(rpr1)
            rpr2 = torch.tensor(rpr2)
        else:   # use geneformer embs
            rpr1 = torch.tensor(self.gene_rpr_map[cancer][data_single[0]])
            rpr2 = torch.tensor(self.gene_rpr_map[cancer][data_single[1]])
        
        label = torch.tensor(data_single[2])    # SL label
        
        #####################################################
        if kg_sentence:
                
            # KG embedding mapping for rpr1 and rpr2
            kg_emb1 = []
            kg_emb2 = []

            for gene_id in rpr1:
                gene_id = gene_id.item()  # Convert tensor to int
                if gene_id in geneid2kgemb:
                    kg_emb1.append(torch.tensor(geneid2kgemb[gene_id]))
                else:
                    # print(f"Missing KG embedding for gene ID: {gene_id}")
                    kg_emb1.append(torch.zeros(kg_emb_size))  # Placeholder for missing embeddings

            for gene_id in rpr2:
                gene_id = gene_id.item()  # Convert tensor to int
                if gene_id in geneid2kgemb:
                    kg_emb2.append(torch.tensor(geneid2kgemb[gene_id]))
                else:
                    # print(f"Missing KG embedding for gene ID: {gene_id}")
                    kg_emb2.append(torch.zeros(kg_emb_size))  # Placeholder for missing embeddings

            # Convert to tensors
            kg1 = torch.stack(kg_emb1)
            kg2 = torch.stack(kg_emb2)
        #####################################################
        
        if self.bi_rpr:
            # map sentence to embeddings
            if self.emb_mtx is not None:
                rpr1 = F.embedding(rpr1, self.emb_mtx).to(torch.float32)
                rpr2 = F.embedding(rpr2, self.emb_mtx).to(torch.float32)

            if self.sent_mask is not None:
                mask1 = self.sent_mask[cancer][data_single[0]][:self.n]
                mask2 = self.sent_mask[cancer][data_single[1]][:self.n]

                if self.augmentation is not None and random.uniform(0,1) > 1/self.augment_fold: # otherwise remain unchanged
                    if self.augmentation == "mask":
                        sent_len1 = mask1.count(1)
                        sent_len2 = mask2.count(1)
                        
                        if sent_len1 > 2:
                            mask_idx1 = get_mask_idx(sent_len1, mask_times=5)
                            for i in mask_idx1:
                                mask1[i] = 0
                        if sent_len2 > 2:
                            mask_idx2 = get_mask_idx(sent_len2, mask_times=5)
                            for i in mask_idx2:
                                mask2[i] = 0
                
                mask1 = torch.tensor(mask1)
                mask2 = torch.tensor(mask2)
                
                #####################################################
                if kg_sentence:
                    # Concatenate the original embeddings with the KG embeddings
                    rpr1 = torch.cat((rpr1, kg1), dim=-1)  # [10, 256 + 128]
                    rpr2 = torch.cat((rpr2, kg2), dim=-1)  # [10, 256 + 128]
                    
                    # rpr1 = kg1  
                    # rpr2 = kg2  
                    
                #####################################################
                # print("return", rpr1.shape, mask1.shape, kg1.shape) # torch.Size([10, 256]) torch.Size([10]) torch.Size([10, 128])
                # print("dataset", data_single[0], data_single[0].dtype) # 5748 int64
                return rpr1, mask1, rpr2, mask2, label, data_single[0], data_single[1], cancer
            else:
                return rpr1, rpr2, label, data_single[0], data_single[1], cancer
        
        else:
            rpr = torch.cat([rpr1, rpr2], dim=0)
            # rpr = np.concatenate((rpr1, rpr2), axis=0)
            return rpr, label, data_single[0], data_single[1], cancer
    


class Emb_Dataset(Dataset):

    def __init__(self, gene1_emb, gene2_emb):

        self.cell_idx = list(gene1_emb.keys())
        self.gene1_emb = list(gene1_emb.values())
        self.gene2_emb = list(gene2_emb.values())

    def __len__(self):

        return len(self.gene1_emb)
    
    def __getitem__(self, idx):

        g1_emb = self.gene1_emb[idx].reshape(1,-1)
        g2_emb = self.gene2_emb[idx].reshape(1,-1)

        emb = np.squeeze(np.concatenate((g1_emb, g2_emb), axis=1))

        cell_idx = self.cell_idx[idx]

        return torch.tensor(emb), cell_idx



class GeneSentenceDataset(Dataset):

    def __init__(self, dataset, emb_mtx, n, gene2anno_map, random_init):

        self.dataset = dataset
        self.emb_mtx = torch.tensor(emb_mtx)
        self.n = n
        if n > 200:
            raise Exception("Please set a valid gene sentence length (<=200)")
        self.gene2anno_map = gene2anno_map
        self.random_init = random_init

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        input_ids = torch.tensor(self.dataset[idx]['input_ids'][1:self.n])  #remove root gene
        # emb = F.embedding(input_ids, self.emb_mtx).to(torch.float32)
        emb = F.embedding(input_ids, self.emb_mtx).to(torch.float32) ## input_ids: gene id
        att_mask = torch.tensor(self.dataset[idx]['attention_mask'][1:self.n])
        cancer = torch.tensor(self.dataset[idx]['cancer'])
        root_gene = torch.tensor(self.dataset[idx]['root_gene'])

        # return emb, att_mask, root_gene, cancer
        # return emb, att_mask, cancer, root_gene
        if self.dataset[idx]['root_gene'] in self.gene2anno_map:
            anno = torch.tensor(self.gene2anno_map[self.dataset[idx]['root_gene']])
        else:
            anno = torch.tensor(0)  # unknown

        if not self.random_init:
            return emb, att_mask, anno, root_gene
        else:
            return input_ids, att_mask, anno, root_gene
    


def get_gene_sent_map(dataset, return_mask=True):

    gene_sent_map = {}
    sent_mask_map = {}

    for i in range(len(dataset)):
        root_gene = dataset[i]['root_gene']
        input_ids = dataset[i]['input_ids']
        att_mask = dataset[i]['attention_mask']
        context = dataset[i]['cancer']

        if context not in gene_sent_map:
            gene_sent_map[context] = {}
            sent_mask_map[context] = {}
        if root_gene not in gene_sent_map[context]:
            gene_sent_map[context][root_gene] = input_ids
            sent_mask_map[context][root_gene] = att_mask

    if return_mask:
        return gene_sent_map, sent_mask_map
    else:
        return gene_sent_map    