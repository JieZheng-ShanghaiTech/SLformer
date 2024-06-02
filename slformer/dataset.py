import torch
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F


class SL_Dataset(Dataset):

    def __init__(self, data, gene_rpr_map, sent_mask=None, emb_mtx=None, bi_rpr=False):

        self.data = data
        self.gene_rpr_map = gene_rpr_map
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

        rpr1 = torch.tensor(self.gene_rpr_map[cancer][data_single[0]])
        rpr2 = torch.tensor(self.gene_rpr_map[cancer][data_single[1]])
        
        label = torch.tensor(data_single[2])    # SL label

        if self.bi_rpr:
            # map sentence to embeddings
            if self.emb_mtx is not None:
                rpr1 = F.embedding(rpr1, self.emb_mtx).to(torch.float32)
                rpr2 = F.embedding(rpr2, self.emb_mtx).to(torch.float32)

            if self.sent_mask is not None:
                mask1 = torch.tensor(self.sent_mask[cancer][data_single[0]])
                mask2 = torch.tensor(self.sent_mask[cancer][data_single[1]])
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

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        input_ids = torch.tensor(self.dataset[idx]['input_ids'])
        att_mask = torch.tensor(self.dataset[idx]['attention_mask'])
        label = torch.tensor(self.dataset[idx]['cancer'])
        root_gene = torch.tensor(self.dataset[idx]['root_gene'])

        return input_ids, att_mask, label, root_gene
    

    def get_gene_sent_map(self, return_mask=True):

        gene_sent_map = {}
        sent_mask_map = {}

        for i in range(len(self.dataset)):
            root_gene = self.dataset[i]['root_gene']
            input_ids = self.dataset[i]['input_ids']
            att_mask = self.dataset[i]['attention_mask']
            context = self.dataset[i]['cancer']

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