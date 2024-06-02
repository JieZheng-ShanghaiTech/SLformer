import numpy as np
from sklearn import metrics
from scipy import stats
import pandas as pd
import pickle as pkl
import os
import torch
import time

from util import set_seed, create_dir, ndcg, ndcg_bin, calc_random_auc, get_train_test_SL, transfer_data_idx, average_metrics, clear_log
from train import train
from model import MLP, Transformer_Finetuner
from dataloader import load_train_data_SL




class Validation_Experiment():

    def __init__(self, config, args, common_data):

        self.config = config
        self.args = args
        self.model_class= config.model_type
        self.experiment = config.task.type

        self.geneformer_emb_map = common_data["geneformer_emb_map"]
        self.geneformer_emb_mtx = common_data["geneformer_emb_mtx"]

        self.gene_sent_map = common_data["gene2sent_map"]
        self.sent_mask_map = common_data["sent_mask_map"]

        self.gene2id_map = common_data["gene2id_map"]
        self.cancer_list = common_data["cancer_list"]
        self.id2cancer_map = common_data["id2cancer_map"]
        # self.cancer2id_map = common_data["cancer2id_map"]

    
    def config_transformer(self):

        # GeneSentence input config =========================
        # pretrained_emb= torch.tensor(self.geneformer_emb_mtx)
        # self.pretrained_emb = pretrained_emb.to(torch.float32)

        # args
        self.transformer_config = {
            "d_model": self.args.d_model,
            "n_head": self.args.n_head,
            "dropout": self.args.dropout,
            "vocab_size": len(self.gene2id_map),
            "dim_feedforward": self.args.dim_feedforward,
            "num_layers": self.args.num_layers,
            "mlp_hidden_dim": self.args.hidden_dim,
            "add_att": self.args.add_att,
            "att_nhead":self.args.att_nhead,
            "freeze_transformer_encoder": False,
        }


    def run_experiment(self, save_model=False, save_log=True):

        if self.model_class == 'transformer':
            self.config_transformer()

        criterion = torch.nn.BCELoss()
        m = torch.nn.Sigmoid()

        # experiment_dir = os.path.join(self.config.EXPERIMENT_DIR, time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()))
        experiment_dir = os.path.join(self.config.EXPERIMENT_DIR, f"{self.experiment}_{self.model_class}_n{self.args.n}")
        log_root_dir = os.path.join(experiment_dir, "log")
        model_root_dir = os.path.join(experiment_dir, "model")
        create_dir(experiment_dir)
        if save_log:
            create_dir(log_root_dir)
        if save_model:
            create_dir(model_root_dir)

        if self.experiment == 'cancer_specific' or self.experiment == 'mix':

            for cancer_type in self.config.task.cancer:
            
                log_path = os.path.join(log_root_dir, f"train_log_{cancer_type}.csv")
                clear_log(log_path)
                
                for cv in range(1,6):
                    data_test = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"test_{cancer_type}_fold_{cv}.npy"))
                    data_train = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"train_{cancer_type}_fold_{cv}.npy"))
                    
                    model_save_path = os.path.join(model_root_dir, f"model_{cancer_type}_cv{cv}.pth")

                    if self.model_class == 'geneformer':
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.geneformer_emb_map, self.args.batch_size)
                        model = MLP(num_layers=2, input_dim=self.args.input_dim, hidden_dim=self.args.hidden_dim, output_dim=self.args.output_dim)
                    elif self.model_class == 'transformer':
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.gene_sent_map, self.args.batch_size, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx)
                        model = Transformer_Finetuner(config=self.transformer_config)

                    train(self.args.device, model, criterion, m, self.args, train_loader, model_save_path, log_path, test_loader, save_model=save_model, save_log=save_log, model_class=self.model_class)
                
                # get average results
                average_metrics(log_path)


        elif self.experiment == 'cross_cancer':

            for cancer, cancer_name in self.id2cancer_map.items():

                data_test = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"test_{cancer_name}.npy"))
                data_train = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"train_{cancer_name}.npy"))

                model_save_path = os.path.join(model_root_dir, f"model_transfer_{cancer_name}.pth")

                log_path = os.path.join(log_root_dir, f"train_log_transfer_{cancer_name}.csv")
                clear_log(log_path)

                for s in range(1,6):
                    set_seed(s)

                    if self.model_class == 'geneformer':
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.geneformer_emb_map, self.args.batch_size)
                        model = MLP(num_layers=2, input_dim=self.args.input_dim, hidden_dim=self.args.hidden_dim, output_dim=self.args.output_dim)
                    elif self.model_class == 'transformer':
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.gene_sent_map, self.args.batch_size, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx)
                        model = Transformer_Finetuner(config=self.transformer_config)

                    train(self.args.device, model, criterion, m, self.args, train_loader, model_save_path, log_path, test_loader, save_model=save_model, save_log=save_log, model_class=self.model_class)

                # get average results
                average_metrics(log_path)


    def save_train_test_data(self, data_total, cancer_type):

        save_dir = os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.config.task.type)
        create_dir(save_dir)

        if self.experiment == 'cancer_specific' or self.experiment == 'mix':
            for cv in range(1,6):
                data_test, data_train = get_train_test_SL(data_total, cv=cv, data_all=False, return_idx=False)
                np.save(os.path.join(save_dir, f"test_{cancer_type}_fold_{cv}.npy"), data_test)
                np.save(os.path.join(save_dir, f"train_{cancer_type}_fold_{cv}.npy"), data_train)
        elif self.experiment == 'cross_cancer':
            for cancer, cancer_name in self.id2cancer_map.items():
                data_test, data_train = get_train_test_SL(data_total, data_all=False, split_by_cancer=True, test_cancer=cancer, return_idx=False)
                np.save(os.path.join(save_dir, f"test_{cancer_name}.npy"), data_test)
                np.save(os.path.join(save_dir, f"train_{cancer_name}.npy"), data_train)


    def get_benchmark_data(self, data_total, cancer_type):

        experiment_dir = os.path.join(self.config.EXPERIMENT_DIR, f"{self.experiment}_{self.model_class}_n{self.args.n}")
        create_dir(experiment_dir)
        save_dir = os.path.join(experiment_dir, "train_test_data")
        create_dir(os.path.join(save_dir))

        if self.experiment == 'cancer_specific':
            for cv in range(1,6):
                data_test, data_train = get_train_test_SL(data_total, cv=cv, data_all=False, return_idx=False)
                df_test = transfer_data_idx(data_test, self.gene2id_map, self.id2cancer_map)
                df_train = transfer_data_idx(data_train, self.gene2id_map, self.id2cancer_map)
                df_test.to_csv(os.path.join(save_dir, f"test_{cancer_type}_cv_{cv}.csv"), index=False)
                df_train.to_csv(os.path.join(save_dir, f"train_{cancer_type}_cv_{cv}.csv"), index=False)
        elif self.experiment == 'cross_cancer':
            for cancer, cancer_name in self.id2cancer_map.items():
                data_test, data_train = get_train_test_SL(data_total, data_all=False, split_by_cancer=True, test_cancer=cancer, return_idx=False)
                df_test = transfer_data_idx(data_test, self.gene2id_map, self.id2cancer_map)
                df_train = transfer_data_idx(data_train, self.gene2id_map, self.id2cancer_map)
                df_test.to_csv(os.path.join(save_dir, f"test_transfer_{cancer_name}.csv"), index=False)
                df_train.to_csv(os.path.join(save_dir, f"train_transfer_{cancer_name}.csv"), index=False)



    # def get_random_auc(self, data_total):

    #     if self.experiment == 'cancer_specific':
    #         rand_auc_res = []
    #         rand_aupr_res = []

    #         for cv in range(1,6):
    #             _, test_loader = load_train_data_SL(data_total, self.geneformer_emb_map, self.args.batch_size, cv=cv)
    #             rand_auc, rand_aupr = calc_random_auc(test_loader)
    #             rand_auc_res.append(rand_auc)
    #             rand_aupr_res.append(rand_aupr)

    #         for met, res in {"AUC":rand_auc_res,"AUPR":rand_aupr_res}.items():
    #             mean = np.round(np.mean(res),4)
    #             std = np.round(np.std(res),4)
    #             print(met, str(mean)+" ("+str(std)+")")

    #     elif self.experiment == 'cross_cancer':
    #         for cancer, cancer_name in self.id2cancer_map.items():
    #             rand_auc_res = []
    #             rand_aupr_res = []
    #             for s in range(1,6):
    #                 _, test_loader = load_train_data_SL(data_total, self.geneformer_emb_map, self.args.batch_size, split_by_cancer=True, test_cancer=cancer)
    #                 rand_auc, rand_aupr = calc_random_auc(test_loader)
    #                 rand_auc_res.append(rand_auc)
    #                 rand_aupr_res.append(rand_aupr)

    #             for met, res in {"AUC":rand_auc_res,"AUPR":rand_aupr_res}.items():
    #                 mean = np.round(np.mean(res),4)
    #                 std = np.round(np.std(res),4)
    #                 print(cancer_name, met, str(mean)+" ("+str(std)+")")


    # def infer_primpartner(self, data_fname, output_dir):

    #     data_total = np.load(os.path.join(self.args.data_dir, f"{data_fname}.npy"))

    #     if self.model_class == 'geneformer':
    #         model = MLP(num_layers=2, input_dim=self.args.input_dim, hidden_dim=self.args.hidden_dim, output_dim=self.args.output_dim, use_selayer=False)
    #         loader = load_train_data_SL(data_total, self.geneformer_emb_map, self.args.batch_size, data_all=True)
    #     elif self.model_class == 'transformer':
    #         model = Transformer_Finetuner(embeddings=self.pretrained_emb, config=self.transformer_config)
    #         loader = load_train_data_SL(data_total, self.gene_sent_map, self.args.batch_size, bi_rpr=True, sent_mask=self.sent_mask_map, data_all=True)

    #     for cv in range(1,6):
    #         experiment_name = self.experiment_name+f"_cv{cv}"

    #         predict_res, _, _, primary_gene, partner_gene = SL_test_prim_partner(device=self.args.device, model=model,
    #             pretrain_file=os.path.join(self.args.model_savepath, f"model_SL_{experiment_name}.pth"),
    #             gene2id_file=self.args.gene2id_file,
    #             data_loader=loader)
            
    #         result_df = pd.DataFrame(data=predict_res.reshape(-1,1), columns=["score"])
    #         result_df.insert(loc=0, column='partner_gene', value=partner_gene)
    #         result_df.insert(loc=0, column='primary_gene', value=primary_gene)

    #         result_df = result_df.sort_values(by=["score"],ascending=False)

    #         result_df.to_csv(os.path.join(output_dir, f"{data_fname}_{experiment_name}.csv"))



def SL_test_prim_partner(device, model, pretrain_file, gene2id_file, data_loader):

    with open(gene2id_file, 'rb') as f:
        gene2id_map = pkl.load(f)
    id2gene_map = {i:g for g,i in gene2id_map.items()}

    params_pretrain = torch.load(pretrain_file).state_dict().copy()
    if 'emb.weight' in params_pretrain:
        del params_pretrain['emb.weight']
    model.load_state_dict(params_pretrain, strict=False)
    model = model.to(device)
    model.eval()

    predict_res = []
    true_label = []
    context = []
    partner_gene_name = []
    primary_gene_name = []

    for i, data in enumerate(data_loader):

        if len(data)==5:
            total_emb, label, g1, g2, cancer = data
            total_emb_cuda = torch.autograd.Variable(total_emb.to(device)).to(torch.float32)
            res = model(total_emb_cuda)
        elif len(data)==8:
            sent1, mask1, sent2, mask2, label, g1, g2, cancer = data
            sent1_cuda = sent1.to(device)
            sent2_cuda = sent2.to(device)
            mask1_cuda = mask1.to(device)
            mask2_cuda = mask2.to(device)
            res = model(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda)

        m = torch.nn.Sigmoid()
        res = torch.squeeze(m(res))
        # res = torch.squeeze(res)
        res = res.detach().cpu()

        predict_res.append(res)
        true_label.append(label)
        context.append(cancer)
        partner_gene_name.extend([id2gene_map[g.item()] for g in g2])
        primary_gene_name.extend([id2gene_map[g.item()] for g in g1])
        
    predict_res = torch.cat(predict_res, dim=0)
    true_label = torch.cat(true_label, dim=0)
    context = torch.cat(context, dim=0)

    return predict_res.numpy(), true_label.numpy(), context.numpy(), primary_gene_name, partner_gene_name




class Downstream_evaluate():

    def __init__(self, data, type):

        self.data = data
        self.predict = data["predict"].values
        self.int_predict = np.around(self.predict,0)
        self.true = data["true"].values

        self.predict_pair_ranked = list(data.sort_values(by='predict', ascending=False).index)
        self.true_pair_ranked = list(data.sort_values(by='true', ascending=True).index) # viability of rank

        topk_range = [10, 20, 30, 50, 100]
        self.topk = [k for k in topk_range if k < len(data)]

        self.type = type

    def calc_metrics(self):

        if self.type == "binary":
        
            result = {
                "auc": metrics.roc_auc_score(self.true, self.predict),
                "aupr": metrics.average_precision_score(self.true, self.predict),
                "f1": metrics.f1_score(self.true, self.int_predict),
                "precision": metrics.precision_score(self.true, self.int_predict),
                "recall": metrics.recall_score(self.true, self.int_predict),
                "acc": metrics.accuracy_score(self.true, self.int_predict),
            }

            # ndcg for binary
            true_hit = list(self.data[self.data["true"]==1].index)
            for k in self.topk:
                result["ndcg_bin@"+str(k)], result["hit@"+str(k)] = ndcg_bin(k, self.predict_pair_ranked, true_hit)


        elif self.type == "rank":

            result = {}

            for k in self.topk:
                result["ndcg@"+str(k)], result["hit@"+str(k)] = ndcg(k, self.predict_pair_ranked, self.true_pair_ranked)


        elif self.type == "pos_score":

            result = {
                # "wilcoxon_p": ranksums(true, predict)[1]
                "spearman_r": stats.spearmanr(self.true, self.predict).correlation
            }

            for k in self.topk:
                result["ndcg@"+str(k)], result["hit@"+str(k)] = ndcg(k, self.predict_pair_ranked, self.true_pair_ranked)


        elif self.type == "bi_score":


            result = {
                # "wilcoxon_p": ranksums(true, predict)[1]
                "spearman_r": stats.spearmanr(self.true, self.predict).correlation
            }

            for k in self.topk:
                result["ndcg@"+str(k)], result["hit@"+str(k)] = ndcg(k, self.predict_pair_ranked, self.true_pair_ranked)
        

        return pd.DataFrame(result, index=[0])