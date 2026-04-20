import torch
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle

"""
Create dataloaders
"""
def load_all_data_SL(all_data, gene_rpr_map, batch_size, n=None, bi_rpr=False, sent_mask=None, emb_mtx=None, add_kg=1):

    loader = DataLoader(SL_Dataset(all_data, gene_rpr_map, n=n, bi_rpr=bi_rpr, sent_mask=sent_mask, emb_mtx=emb_mtx, add_kg=add_kg), batch_size=batch_size, shuffle=False)
    
    return loader


def load_train_data_SL(test_data, train_data, gene_rpr_map, batch_size, n=None, bi_rpr=False, sent_mask=None, emb_mtx=None, add_kg=1):

    train_dataset = SL_Dataset(train_data, gene_rpr_map, n=n, bi_rpr=bi_rpr, sent_mask=sent_mask, emb_mtx=emb_mtx, add_kg=add_kg)
    test_dataset = SL_Dataset(test_data, gene_rpr_map, n=n, bi_rpr=bi_rpr, sent_mask=sent_mask, emb_mtx=emb_mtx, add_kg=add_kg)

    drop_last = {"train":False, "test":False}
    for type, dataset in {"train":train_dataset, "test":test_dataset}.items():
        if len(dataset)%batch_size < 20:   # avoid the case that the last batch is too small
            drop_last[type] = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last["train"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last["test"])

    return train_loader, test_loader

"""
Dataset
"""
class SL_Dataset(Dataset):

    def __init__(self, data, gene_rpr_map, n=None, sent_mask=None, emb_mtx=None, bi_rpr=False, add_kg=1):
        """
        Dataset for gene sentences of SL gene pairs
        args:
            - data: npy data of gene pairs with context and SL annotations (see prepare_data.py)
            - gene_rpr_map: mapping from genes to either gene sentence representations or Geneformer embeddings
            - n: gene sentence length (<200)
            - sent_mask: boolean masks of gene sentences
            - emb_mtx: matrix of Geneformer embeddings across all contexts
            - bi_rpr: boolean indicator, whether or not concatening two gene representations, False for concatenating
            - add_kg: integer indicator, whether or not concatening with KG embeddings (0-not concatenate, 1-concatenate, 2-only using KG embeddings)
        """

        self.data = data
        self.gene_rpr_map = gene_rpr_map
        self.n = n
        if n is not None and n > 200:
            raise Exception("Please set a valid gene sentence length (<=200)")

        self.sent_mask = sent_mask
        if emb_mtx is not None:
            self.emb_mtx = torch.tensor(emb_mtx)
        self.bi_rpr = bi_rpr
        self.add_kg = add_kg

        if self.add_kg:
            with open('data/saved_data/map/geneid2kgemb256_p1.pkl', 'rb') as f:
                self.geneid2kgemb = pickle.load(f)
            self.kg_emb_size = 256 # 128, 768

    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, idx):

        data_single = self.data[idx]
        # [g1, g2, label, cancer]
        cancer = data_single[3]

        if self.n is not None:  # use gene sentence
            rpr1 = self.gene_rpr_map[cancer][data_single[0]][:self.n]
            rpr2 = self.gene_rpr_map[cancer][data_single[1]][:self.n]
            rpr1 = torch.tensor(rpr1)
            rpr2 = torch.tensor(rpr2)
        else:   # use geneformer embs
            rpr1 = torch.tensor(self.gene_rpr_map[cancer][data_single[0]])
            rpr2 = torch.tensor(self.gene_rpr_map[cancer][data_single[1]])
        
        label = torch.tensor(data_single[2])    # SL labels
        
        if self.add_kg != 0:
                
            # KG embedding mapping for rpr1 and rpr2
            kg_emb1 = []
            kg_emb2 = []

            for gene_id in rpr1:
                gene_id = gene_id.item()  # Convert tensor to int
                if gene_id in self.geneid2kgemb:
                    kg_emb1.append(torch.tensor(self.geneid2kgemb[gene_id]))
                else:
                    # print(f"Missing KG embedding for gene ID: {gene_id}")
                    kg_emb1.append(torch.zeros(self.kg_emb_size))  # Placeholder for missing embeddings

            for gene_id in rpr2:
                gene_id = gene_id.item()  # Convert tensor to int
                if gene_id in self.geneid2kgemb:
                    kg_emb2.append(torch.tensor(self.geneid2kgemb[gene_id]))
                else:
                    # print(f"Missing KG embedding for gene ID: {gene_id}")
                    kg_emb2.append(torch.zeros(self.kg_emb_size))  # Placeholder for missing embeddings

            # Convert to tensors
            kg1 = torch.stack(kg_emb1)
            kg2 = torch.stack(kg_emb2)


        if not self.bi_rpr:
            if self.add_kg==1:
                # Concatenate the original embeddings with the KG embeddings
                rpr1 = torch.cat((rpr1, kg1[0]), dim=-1)
                rpr2 = torch.cat((rpr2, kg2[0]), dim=-1)
            rpr = torch.cat([rpr1, rpr2], dim=0)
            return rpr, label, data_single[0], data_single[1], cancer

        else:
            # map sentence to embeddings
            if self.emb_mtx is not None:
                rpr1 = F.embedding(rpr1, self.emb_mtx).to(torch.float32)
                rpr2 = F.embedding(rpr2, self.emb_mtx).to(torch.float32)

            if self.sent_mask is not None:
                mask1 = self.sent_mask[cancer][data_single[0]][:self.n]
                mask2 = self.sent_mask[cancer][data_single[1]][:self.n]                
                mask1 = torch.tensor(mask1)
                mask2 = torch.tensor(mask2)
                
                if self.add_kg == 1:
                    # Concatenate the original embeddings with the KG embeddings
                    rpr1 = torch.cat((rpr1, kg1), dim=-1)
                    rpr2 = torch.cat((rpr2, kg2), dim=-1)
                elif self.add_kg == 2:  
                    ## only using KG embedding
                    rpr1 = kg1
                    rpr2 = kg2

                return rpr1, mask1, rpr2, mask2, label, data_single[0], data_single[1], cancer
            else:
                return rpr1, rpr2, label, data_single[0], data_single[1], cancer
    



