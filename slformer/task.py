import numpy as np
from sklearn.metrics import ndcg_score, average_precision_score, precision_recall_curve
import pandas as pd
import os
import pickle as pkl
import torch
import time
import json
import logging
import wandb
from datasets import load_from_disk

from util import create_dir, ndcg, ndcg_bin, mean_metrics, average_metrics, clear_result, precision_at_k, recall_at_k, hit_at_k, hit_at_k_bin
from util import stat_indep_test, get_train_test_SL
from train import train, pretrain
from model import MLP, Transformer_Finetuner, Transformer_Pretrain
from dataloader import load_train_data_SL, load_all_data_SL, load_pretrain_data, load_pretrain_data_all
from prepare_data import filt_SL_test_names, single_permute_data



class Validation_Experiment():

    def __init__(self, config, args, common_data):

        self.config = config
        self.args = args
        self.common_data = common_data
        self.experiment = config.task.type

        self.geneformer_emb_map = common_data["geneformer_emb_map"]
        self.geneformer_emb_mtx = common_data["geneformer_emb_mtx"]

        self.gene_sent_map = common_data["gene2sent_map"]
        self.sent_mask_map = common_data["sent_mask_map"]

        self.gene2id_map = common_data["gene2id_map"]
        self.cancer_list = common_data["cancer_list"]
        self.cancer2id_map = common_data["cancer2id_map"]
        self.id2cancer_map = {i:c for i,c in enumerate(self.cancer_list)}

    
    def config_transformer(self, args):

        # GeneSentence input config =========================
        # pretrained_emb= torch.tensor(self.geneformer_emb_mtx)
        # self.pretrained_emb = pretrained_emb.to(torch.float32)

        # args
        transformer_config = {
            "d_model": args.d_model,
            "n_head": args.n_head,
            "dropout": args.dropout,
            "vocab_size": len(self.gene2id_map),
            "transformer_hidden_dim": args.transformer_hidden_dim,
            "transformer_num_layers": args.transformer_num_layers,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "mlp_output_dim": args.mlp_output_dim,
            "add_att": args.add_att,
            "att_nhead": args.att_nhead,
            "att_num_layers": args.att_num_layers,
            "random_init": args.random_init,
            
            # "freeze_transformer_encoder": False,
        }

        return transformer_config


    def load_pretrain_checkpoint(self, args, config, cv):

        transformer_args = ['n', 'd_model', 'n_head', 'dropout', 'transformer_hidden_dim', 'transformer_num_layers',' random_init']

        if 'mix_checkpoint' in config.task:
            model_dir = config.task.mix_checkpoint.path
            model_savename = f'model_all_cv{cv}.pth'
            with open(os.path.join(model_dir, 'params.json'), 'r') as f:
                model_params = json.load(f)
            for arg in vars(args):
                if arg in model_params and arg in transformer_args:
                    setattr(args, arg, model_params[arg])
            mix_checkpoint=os.path.join(model_dir, "model", model_savename)
            params_pretrain = torch.load(mix_checkpoint)

            return args, params_pretrain

        elif 'pretrain_checkpoint' in config.task:
            model_dir = config.task.pretrain_checkpoint.path
            with open(os.path.join(model_dir, 'params.json'), 'r') as f:
                model_params = json.load(f)
            for arg in vars(args):
                if arg in model_params and arg in transformer_args:
                    setattr(args, arg, model_params[arg])
            transformer_config = self.config_transformer(args)
            model = Transformer_Finetuner(config=transformer_config)
            pretrain_checkpoint = os.path.join(model_dir, "model", "model.pth")
            params_pretrain = torch.load(pretrain_checkpoint)
            new_state_dict = model.state_dict()
            filt_state_dict = {k: v for k, v in params_pretrain.items() if 'predictor' not in k}
            new_state_dict.update(filt_state_dict)

            return args, new_state_dict

    
    def gsent_pretrain(self, save_model=True, save_result=True):

        criterion = torch.nn.CrossEntropyLoss()

        random_init = self.args.random_init

        # self.args.output_dim = len(self.gene2id_map)
        # self.args.output_dim = 9    #num of cancer types
        gene2anno_map = self.common_data["gene2go_map"]
        self.args.output_dim = len(set(gene2anno_map.values()))+1
        transformer_config = self.config_transformer(self.args)

        # save path of experiment results, models, logs, and params   
        curr_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()) 
        experiment_dir = os.path.join(self.config.EXPERIMENT_DIR, "pretrain", curr_time)
        result_root_dir = os.path.join(experiment_dir, "result")
        model_root_dir = os.path.join(experiment_dir, "model")
        log_fp = os.path.join(experiment_dir, "log.txt")
        params_fp = os.path.join(experiment_dir, "params.json")
        
        create_dir(experiment_dir)
        if save_result:
            create_dir(result_root_dir)
        if save_model:
            create_dir(model_root_dir)
        
        file_handler = logging.FileHandler(log_fp, mode='w')
        logging.getLogger().addHandler(file_handler)

        params = {}
        params.update(vars(self.args))
        params.update(self.config)
        with open(params_fp, "w") as f:
            json.dump(params, f, indent=4)
        
        ## Start pretraining
        result_path = os.path.join(result_root_dir, f"train_result.csv")

        gsent_data = load_from_disk("/data/xinliu/GeneSentence/TISCH2/gene_sentence/gene_sentence_n200_sc")
        filt_data = gsent_data.filter(lambda s: s['length'] >= 2)
        dataset_split = filt_data.train_test_split(test_size=0.1, seed=1)
        data_train = dataset_split["train"]
        data_test = dataset_split["test"]

        print("Start pretraining...", f"train data size={len(data_train)}, test data size={len(data_test)}")
        # print("Start pretraining...", f"data size={len(filt_data)}")

        model_save_path = os.path.join(model_root_dir, f"model.pth")
        # train_loader, test_loader = load_pretrain_data(data_train, data_test, batch_size=self.args.batch_size, emb_mtx=self.geneformer_emb_mtx, n=self.args.n, gene2anno_map=gene2anno_map, random_init=random_init)
        data_loader= load_pretrain_data_all(filt_data, batch_size=self.args.batch_size, emb_mtx=self.geneformer_emb_mtx, n=self.args.n, gene2anno_map=gene2anno_map, random_init=random_init)
        model = Transformer_Pretrain(config=transformer_config)

        # pretrain(self.args.device, model, criterion, self.args, train_loader, test_loader, model_save_path, result_path, save_model=save_model, save_result=save_result)
        pretrain(self.args.device, model, criterion, self.args, data_loader, model_save_path, result_path, save_model=save_model, save_result=save_result)


    def fewshot_train(self):

        criterion = torch.nn.BCELoss()
        m = torch.nn.Sigmoid()
        # save path of experiment results, models, logs, and params   
        curr_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()) 
        experiment_dir = os.path.join(self.config.EXPERIMENT_DIR, f"{self.experiment}", curr_time)
        result_root_dir = os.path.join(experiment_dir, "result")
        model_root_dir = os.path.join(experiment_dir, "model")
        log_fp = os.path.join(experiment_dir, "log.txt")
        params_fp = os.path.join(experiment_dir, "params.json")

        create_dir(result_root_dir)
        create_dir(model_root_dir)

        file_handler = logging.FileHandler(log_fp, mode='w')
        logging.getLogger().addHandler(file_handler)

        params = {}
        params.update(vars(self.args))
        params.update(self.config)
        with open(params_fp, "w") as f:
            json.dump(params, f, indent=4)

        cancer_out = self.config.task.cancer_out
        result_path = os.path.join(result_root_dir, f"train_result_{cancer_out}.csv")

        for cv in range(1,6):
            data_test = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_fewshot_data", f"test_{cancer_out}_fold_{cv}.npy"))
            data_train = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_fewshot_data", f"train_{cancer_out}_fold_{cv}.npy"))

            print(f"{cancer_out}_out_cv{cv}, train data size={len(data_train)}, test data size={len(data_test)}")

            model_save_path = os.path.join(model_root_dir, f"model_{cancer_out}_cv{cv}.pth")
            train_loader, test_loader = load_train_data_SL(data_test, data_train, self.gene_sent_map, self.args.batch_size, self.args.n, self.args.anchor, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg, augmentation=self.args.augmentation)
            transformer_config = self.config_transformer(self.args)
            model = Transformer_Finetuner(config=transformer_config)

            train(self.args.device, model, criterion, m, self.args, train_loader, model_save_path, result_path, test_loader, save_model=1, save_result=1, model_class='transformer')
        
        average_metrics(result_path)


    def fewshot_test(self, test='mix_fewshot'):
        
        cancer_out = self.config.task.cancer_out

        data_test = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_fewshot_data", f"fewshot_{cancer_out}_val.npy"))
        data_train = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_fewshot_data", f"fewshot_{cancer_out}_train.npy"))
        print(f"{cancer_out}_out_fewshot, train data size={len(data_train)}, test data size={len(data_test)}")

        train_loader, val_loader = load_train_data_SL(data_test, data_train, self.gene_sent_map, self.args.batch_size, self.args.n, self.args.anchor, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg, augmentation=self.args.augmentation)

        ## using mix checkpoints
        if test == 'mix_fewshot':
            # load pretrained model
            model_dir = self.config.task.mix_checkpoint
            # with open(os.path.join(model_dir, 'params.json'), 'r') as f:
            #     model_params = json.load(f)
            # for arg in vars(self.args):
            #     if arg in model_params:
            #         setattr(self.args, arg, model_params[arg])
            transformer_config = self.config_transformer(self.args)

            ## direct inference
            for model_savename in os.listdir(os.path.join(model_dir, "model")):
                model = Transformer_Finetuner(config=transformer_config)    
                mix_checkpoint=os.path.join(model_dir, "model", model_savename)

                predict_res, true_label, cancer, primary_gene, partner_gene = SL_test_prim_partner(device=self.args.device, model=model,
                    pretrain_file=mix_checkpoint,
                    data_loader=val_loader,
                    gene2id_map=self.gene2id_map,
                    cancer2id_map=self.cancer2id_map
                )
                pred_result = pd.DataFrame(data=np.concatenate((np.array(primary_gene).reshape(-1,1),
                                                            np.array(partner_gene).reshape(-1,1),
                                                            np.array(cancer).reshape(-1,1),
                                                            predict_res.reshape(-1,1),
                                                            true_label.reshape(-1,1)), axis=1),
                                                            columns = ["gene1","gene2","cancer","predict","true"])
                pred_result = pred_result.astype({"gene1":'str',"gene2":'str',"cancer":'str',"predict":'float32',"true":'float32'})
                ## eval
                AUPR =  average_precision_score(true_label, predict_res)
                precision, recall, _ = precision_recall_curve(true_label, predict_res)
                F1 = max(2 * precision * recall / (precision + recall))
                print(f"{test} - AUPR:{AUPR:.4g} - F1:{F1:.4g}")


            ## fewshot training
            # result_path = os.path.join(result_root_dir, f"train_result.csv")
            # for model_savename in os.listdir(os.path.join(model_dir, "model")):
            #     model_save_path = os.path.join(model_root_dir, model_savename)
            #     model = Transformer_Finetuner(config=transformer_config)    
            #     mix_checkpoint=os.path.join(model_dir, "model", model_savename)
            #     params_pretrain = torch.load(mix_checkpoint)
            #     model.load_state_dict(params_pretrain, strict=True)
                
            #     train(self.args.device, model, criterion, m, self.args, train_loader, model_save_path, result_path, test_loader, save_model=1, save_result=1, model_class='transformer')

            # average_metrics(result_path)


        ## cancer-specific (trained from scratch)
        elif test == 'cancer_specific':

            criterion = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            curr_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()) 
            experiment_dir = os.path.join(self.config.EXPERIMENT_DIR, f"{self.experiment}", curr_time)
            result_root_dir = os.path.join(experiment_dir, "result")
            model_root_dir = os.path.join(experiment_dir, "model")
            log_fp = os.path.join(experiment_dir, "log.txt")
            params_fp = os.path.join(experiment_dir, "params.json")
            result_path = os.path.join(result_root_dir, f"train_result.csv")

            create_dir(result_root_dir)
            create_dir(model_root_dir)

            file_handler = logging.FileHandler(log_fp, mode='w')
            logging.getLogger().addHandler(file_handler)

            params = {}
            params.update(vars(self.args))
            params.update(self.config)
            params.update({'test': test})
            with open(params_fp, "w") as f:
                json.dump(params, f, indent=4)
            
            ## training and selecting models
            for cv in range(1,6):
                data_out_test, data_out_train = get_train_test_SL(data_train, test_size=0.2, cv=cv, data_all=False, return_idx=False)
                print(len(data_out_test), len(data_out_train))
                train_loader, test_loader = load_train_data_SL(data_out_test, data_out_train, self.gene_sent_map, self.args.batch_size, self.args.n, self.args.anchor, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg, augmentation=self.args.augmentation)
            
                model_save_path = os.path.join(model_root_dir, f"model_cv{cv}.pth")
                transformer_config = self.config_transformer(self.args)
                model = Transformer_Finetuner(config=transformer_config)

                train(self.args.device, model, criterion, m, self.args, train_loader, model_save_path, result_path, test_loader, save_model=1, save_result=1, model_class='transformer')
                
            ## direct inference
            for model_savename in os.listdir(model_root_dir):
                model = Transformer_Finetuner(config=transformer_config)    

                predict_res, true_label, cancer, primary_gene, partner_gene = SL_test_prim_partner(device=self.args.device, model=model,
                    pretrain_file=os.path.join(model_root_dir, model_savename),
                    data_loader=val_loader,
                    gene2id_map=self.gene2id_map,
                    cancer2id_map=self.cancer2id_map
                )
                pred_result = pd.DataFrame(data=np.concatenate((np.array(primary_gene).reshape(-1,1),
                                                            np.array(partner_gene).reshape(-1,1),
                                                            np.array(cancer).reshape(-1,1),
                                                            predict_res.reshape(-1,1),
                                                            true_label.reshape(-1,1)), axis=1),
                                                            columns = ["gene1","gene2","cancer","predict","true"])
                pred_result = pred_result.astype({"gene1":'str',"gene2":'str',"cancer":'str',"predict":'float32',"true":'float32'})
                ## eval
                AUPR =  average_precision_score(true_label, predict_res)
                precision, recall, _ = precision_recall_curve(true_label, predict_res)
                F1 = max(2 * precision * recall / (precision + recall))
                print(f"{test} - AUPR:{AUPR:.4g} - F1:{F1:.4g}")        


        ## cross-cancer (directly tested on the hold-out validation data)
        elif test == 'cross_cancer':
            pass
    

    def finetune_GBM(self):

        train_data = np.load("./data/saved_data/GBM_SL/GBM_SL.npy")
        print(f"GBM finetuning, train data size={len(train_data)}")
        train_loader = load_all_data_SL(train_data, self.gene_sent_map, self.args.batch_size, self.args.n, self.args.anchor, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg, augmentation=self.args.augmentation)
        
        model_dir = self.config.task.mix_checkpoint
        with open(os.path.join(model_dir, 'params.json'), 'r') as f:
            model_params = json.load(f)
        for arg in vars(self.args):
            if arg in model_params and 'lr' not in arg and 'epoch' not in arg: ## use new lr and n_epoch
                setattr(self.args, arg, model_params[arg])
        transformer_config = self.config_transformer(self.args)
        
        criterion = torch.nn.BCELoss()
        m = torch.nn.Sigmoid()
        curr_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()) 
        experiment_dir = os.path.join(self.config.EXPERIMENT_DIR, f"{self.experiment}", curr_time)
        result_root_dir = os.path.join(experiment_dir, "result")
        model_root_dir = os.path.join(experiment_dir, "model")
        log_fp = os.path.join(experiment_dir, "log.txt")
        params_fp = os.path.join(experiment_dir, "params.json")
        
        result_path = os.path.join(result_root_dir, f"train_result.csv")

        create_dir(result_root_dir)
        create_dir(model_root_dir)

        file_handler = logging.FileHandler(log_fp, mode='w')
        logging.getLogger().addHandler(file_handler)

        params = {}
        params.update(vars(self.args))
        params.update(self.config)
        with open(params_fp, "w") as f:
            json.dump(params, f, indent=4)

        for model_savename in os.listdir(os.path.join(model_dir, "model")):
            model_save_path = os.path.join(model_root_dir, model_savename)
            model = Transformer_Finetuner(config=transformer_config)    
            mix_checkpoint=os.path.join(model_dir, "model", model_savename)
            params_pretrain = torch.load(mix_checkpoint)
            model.load_state_dict(params_pretrain, strict=True)
            
            train(self.args.device, model, criterion, m, self.args, train_loader, model_save_path, result_path, save_model=1, save_result=1, model_class='transformer')


    def run_experiment(self, save_model=False, save_result=True, wandb_track=False):
        
        if wandb_track:
            wandb.init()
            for arg in vars(self.args):
                if hasattr(wandb.config, arg):
                    setattr(self.args, arg, getattr(wandb.config, arg))
        
        model_class= self.config.model_type
        if model_class == 'transformer':
            transformer_config = self.config_transformer(self.args)

        criterion = torch.nn.BCELoss()
        # pos_weight = torch.tensor([10.0]).to(device=torch.device("cuda:" + str(self.args.device)))
        # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = torch.nn.CrossEntropyLoss()
        m = torch.nn.Sigmoid()

        # save path of experiment results, models, logs, and params   
        curr_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()) 
        experiment_dir = os.path.join(self.config.EXPERIMENT_DIR, f"{self.experiment}", f"{model_class}", curr_time)
        result_root_dir = os.path.join(experiment_dir, "result")
        model_root_dir = os.path.join(experiment_dir, "model")
        log_fp = os.path.join(experiment_dir, "log.txt")
        params_fp = os.path.join(experiment_dir, "params.json")
        
        create_dir(experiment_dir)
        if save_result:
            create_dir(result_root_dir)
        if save_model:
            create_dir(model_root_dir)
        
        file_handler = logging.FileHandler(log_fp, mode='w')
        logging.getLogger().addHandler(file_handler)

        params = {}
        params.update(vars(self.args))
        params.update(self.config)
        with open(params_fp, "w") as f:
            json.dump(params, f, indent=4)

        # Start experiment
            
        if self.experiment == 'cancer_specific' or self.experiment == 'cancer_specific_all':

            # wandb setting
            if wandb_track:
                run = wandb.init(group=f"{self.experiment}", name=f"{self.experiment}_{curr_time}", reinit=True)
            else:
                run = None
            
            for cancer_type in self.config.task.cancer:
            
                result_path = os.path.join(result_root_dir, f"train_result_{cancer_type}.csv")
                # clear_result(result_path)
                
                for cv in range(1,6):
                    data_test = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"test_{cancer_type}_fold_{cv}.npy"))
                    data_train = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"train_{cancer_type}_fold_{cv}.npy"))

                    print(f"{cancer_type}_cv{cv}, train data size={len(data_train)}, test data size={len(data_test)}")
                    
                    model_save_path = os.path.join(model_root_dir, cancer_type, f"model_{cancer_type}_cv{cv}.pth")
                    create_dir(os.path.join(model_root_dir, cancer_type))

                    if model_class == 'geneformer':
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.geneformer_emb_map, self.args.batch_size, add_kg=self.args.add_kg)
                        model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                    elif model_class == 'transformer':
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.gene_sent_map, self.args.batch_size, self.args.n, self.args.anchor, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg, augmentation=self.args.augmentation)
                        if 'mix_checkpoint' in self.config.task or 'pretrain_checkpoint' in self.config.task:
                            ckp_args, ckp = self.load_pretrain_checkpoint(self.args, self.config, cv)
                            transformer_config = self.config_transformer(ckp_args)
                            model = Transformer_Finetuner(config=transformer_config)
                            model.load_state_dict(ckp, strict=True)
                        else:
                            transformer_config = self.config_transformer(self.args)
                            model = Transformer_Finetuner(config=transformer_config) # config=self.transformer_config

                    train(self.args.device, model, criterion, m, self.args, train_loader, model_save_path, result_path, test_loader, save_model=save_model, save_result=save_result, model_class=model_class, wandb_run=run)

                # get average results
                if wandb_track:
                    avg_metrics = mean_metrics(result_path)
                    run.log(avg_metrics)

                if save_result:
                    average_metrics(result_path)

            if wandb_track: 
                run.finish()
        


        # if self.experiment == 'mix' or self.experiment == 'mix_all':
        # if self.experiment == 'mix' or self.experiment == 'mix_all' or self.experiment=='mix_add_GBM' or self.experiment=='mix_add_GBM_stratified':
        if 'mix' in self.experiment:
            # wandb setting
            if wandb_track:
                run = wandb.init(group=f"{self.experiment}", name=f"{self.experiment}_{curr_time}", reinit=True)
            else:
                run = None

            for cancer_type in self.config.task.cancer:
            
                result_path = os.path.join(result_root_dir, f"train_result_{cancer_type}.csv")
                # clear_result(result_path)
                
                for cv in range(1,6):
                    data_test = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"test_{cancer_type}_fold_{cv}.npy"))
                    data_train = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"train_{cancer_type}_fold_{cv}.npy"))

                    print(f"{cancer_type}_cv{cv}, train data size={len(data_train)}, test data size={len(data_test)}")
                    
                    model_save_path = os.path.join(model_root_dir, f"model_{cancer_type}_cv{cv}.pth")
                    
                    if model_class == 'geneformer':
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.geneformer_emb_map, self.args.batch_size, add_kg=self.args.add_kg)
                        model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                    elif model_class == 'transformer':
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.gene_sent_map, self.args.batch_size, self.args.n, self.args.anchor, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg, augmentation=self.args.augmentation)
                        if 'pretrain_checkpoint' in self.config.task:
                            ckp_args, ckp = self.load_pretrain_checkpoint(self.args, self.config, cv)
                            transformer_config = self.config_transformer(ckp_args)
                            model = Transformer_Finetuner(config=transformer_config)
                            model.load_state_dict(ckp, strict=True)
                        else:
                            transformer_config = self.config_transformer(self.args)
                            model = Transformer_Finetuner(config=transformer_config)

                    
                    train(self.args.device, model, criterion, m, self.args, train_loader, model_save_path, result_path, test_loader, save_model=save_model, save_result=save_result, model_class=model_class, wandb_run=run)

                # get average results
                if wandb_track:
                    avg_metrics = mean_metrics(result_path)
                    run.log(avg_metrics)

                if save_result:
                    average_metrics(result_path)

            if wandb_track: 
                run.finish()


        # elif self.experiment == 'cross_cancer':
        elif self.experiment == 'cross_cancer' or self.experiment == 'cross_cancer_all':

            result_path = os.path.join(result_root_dir, f"train_result_cross_cancer.csv")
            # clear_result(result_path)

            if wandb_track:
                run = wandb.init(group=f"{self.experiment}", name=f"{self.experiment}_{curr_time}", reinit=True)
            else:
                run = None

            # for cancer, cancer_name in self.id2cancer_map.items():
            for cancer_name in self.config.task.cancer:

                model_save_path = os.path.join(model_root_dir, f"model_transfer2{cancer_name}.pth")

                data_test = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"test_{cancer_name}.npy"))
                data_train = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"train_{cancer_name}.npy"))

                print(f"{cancer_name}, train data size={len(data_train)}, test data size={len(data_test)}")

                if model_class == 'geneformer':
                    train_loader, test_loader = load_train_data_SL(data_test, data_train, self.geneformer_emb_map, self.args.batch_size, add_kg=self.args.add_kg)
                    model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                elif model_class == 'transformer':
                    train_loader, test_loader = load_train_data_SL(data_test, data_train, self.gene_sent_map, self.args.batch_size, self.args.n, self.args.anchor, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg, augmentation=self.args.augmentation)
                    if 'pretrain_checkpoint' in self.config.task:
                        origin_model = Transformer_Finetuner(config=self.transformer_config)
                        ckp_args, ckp = self.load_pretrain_checkpoint(self.args, self.config, cv)
                        transformer_config = self.config_transformer(ckp_args)
                        model = Transformer_Finetuner(config=transformer_config)
                        model.load_state_dict(ckp, strict=True)
                    else:
                        transformer_config = self.config_transformer(self.args)
                        model = Transformer_Finetuner(config=transformer_config)

                train(self.args.device, model, criterion, m, self.args, train_loader, model_save_path, result_path, test_loader, save_model=save_model, save_result=save_result, model_class=model_class, wandb_run=run)

            # get average results
            if wandb_track:
                avg_metrics = mean_metrics(result_path)
                run.log(avg_metrics)
                run.finish()

            if save_result:
                average_metrics(result_path)
            

    def independent_test(self, stat=False):

        if self.experiment != "independent_test":
            raise Exception("Please set independent test configs!")

        output_dir = os.path.join(self.config.EXPERIMENT_DIR, self.experiment)
        create_dir(output_dir)
        pred_dir = os.path.join(output_dir, "pred")
        eval_dir = os.path.join(output_dir, "eval")
        eval_cv_dir = os.path.join(output_dir, "eval_cv")
        
        test_datasets = self.config.SL_dataset
        model_dirs = self.config.task.model

        for data_type in list(test_datasets.keys()):
            if data_type != "general":
                context = test_datasets[data_type]["context"]

                model_cfg = model_dirs[context]

                # for scenario, model_cfg in model_dirs.items():
                create_dir(os.path.join(pred_dir, context))
                create_dir(os.path.join(eval_dir, context))
                create_dir(os.path.join(eval_cv_dir, context))

                ## set model parameters
                with open(os.path.join(model_cfg.path, 'params.json'), 'r') as f:
                    model_params = json.load(f)
                for arg in vars(self.args):
                    if arg in model_params:
                        setattr(self.args, arg, model_params[arg])

                transformer_config = self.config_transformer(self.args)

                data = np.load(os.path.join(self.config.SAVED_DATA_DIR, "independent_test_data", f"{data_type}.npy"))
                if model_cfg.model_type == 'geneformer':
                    loader = load_all_data_SL(data, self.geneformer_emb_map, self.args.batch_size, add_kg=self.args.add_kg)
                elif model_cfg.model_type == 'transformer':
                    loader = load_all_data_SL(data, self.gene_sent_map, self.args.batch_size, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg)

                df_all = []
                if context == 'mix':
                    model_dir = os.path.join(model_cfg.path, "model")
                else:
                    model_dir = os.path.join(model_cfg.path, "model", context)

                for i, model_savepath in enumerate(os.listdir(model_dir)):
            
                    if model_cfg.model_type == 'geneformer':
                        model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                    elif model_cfg.model_type == 'transformer':
                        model = Transformer_Finetuner(config=transformer_config)
                    
                    predict_res, true_label, cancer, primary_gene, partner_gene = SL_test_prim_partner(device=self.args.device, model=model,
                        pretrain_file=os.path.join(model_dir, model_savepath),
                        data_loader=loader,
                        gene2id_map=self.gene2id_map,
                        cancer2id_map=self.cancer2id_map
                    )
                    pred_result = pd.DataFrame(data=np.concatenate((np.array(primary_gene).reshape(-1,1),
                                                                np.array(partner_gene).reshape(-1,1),
                                                                np.array(cancer).reshape(-1,1),
                                                                predict_res.reshape(-1,1),
                                                                true_label.reshape(-1,1)), axis=1),
                                                                columns = ["gene1","gene2","cancer","predict","true"])
                    pred_result = pred_result.astype({"gene1":'str',"gene2":'str',"cancer":'str',"predict":'float32',"true":'float32'})
                    pred_result.to_csv(os.path.join(pred_dir, context, f"{data_type}_{i}.csv"), index=False)
                    label_type = test_datasets[data_type].label_type
                    eval_result = independent_evaluate(pred_result, label_type)
                    df_all.append(eval_result)
                
                df_concat = pd.concat(df_all)
                avg_mean = np.round(df_concat.mean(), 4)
                avg_std = np.round(df_concat.std(), 4)
                # avg_results = pd.concat([pd.DataFrame([avg_mean]), pd.DataFrame([avg_std])], axis=0)
                # avg_results.index = ['mean','std']
                avg_results = pd.DataFrame([str(avg_mean[i])+" ("+str(avg_std[i])+")" for i in range(len(avg_mean))]).T
                avg_results.columns = df_concat.columns

                avg_results.to_csv(os.path.join(eval_dir, context, f"{data_type}_{model_cfg.model_type}.csv"), index=False)
                df_concat.to_csv(os.path.join(eval_cv_dir, context, f"{data_type}_{model_cfg.model_type}.csv"), index=False)

        if stat:
            perform_diff_cancer_specific = stat_indep_test(task="cancer_specific")
            perform_diff_mix = stat_indep_test(task="mix")

            print("cancer-specific model:", self.config.task.model.mix.path)
            print("mixed-cancer model:", self.config.task.model.LAML.path)

            print("perform_diff_cancer_specific=", perform_diff_cancer_specific)
            print("perform_diff_mix=", perform_diff_mix)


    def independent_test_on_mix(self, save_raw=False, raw_score=False, random_simu=False):

        self.experiment = "independent_test_on_mix"
        if save_raw:
            if raw_score:
                output_dir = os.path.join(self.config.EXPERIMENT_DIR, self.experiment, 'raw_score')
            else:
                output_dir = os.path.join(self.config.EXPERIMENT_DIR, self.experiment, 'raw')
        else:
            output_dir = os.path.join(self.config.EXPERIMENT_DIR, self.experiment)
        create_dir(output_dir)

        test_datasets = self.config.SL_dataset
        model_dirs = self.config.task.model

        expr_model_type = ""
        data_save = []

        for data_type in list(test_datasets.keys()):
            if data_type != "general" and test_datasets[data_type]["context"]=="mix":

                context_all = test_datasets[data_type]["context_all"]
                
                ## cancer-specific model
                df_cancer_specific = {i:[] for i in range(5)}
                for context in context_all:

                    model_cfg = model_dirs[context]

                    ## set model parameters
                    with open(os.path.join(model_cfg.path, 'params.json'), 'r') as f:
                        model_params = json.load(f)
                    for arg in vars(self.args):
                        if arg in model_params:
                            setattr(self.args, arg, model_params[arg])

                    transformer_config = self.config_transformer(self.args)

                    data = np.load(os.path.join(self.config.SAVED_DATA_DIR, "independent_test_data", f"{data_type}.npy"))
                    ## filt context-specific data
                    context_id = self.cancer2id_map[context]
                    data_context = data[data[:, 3] == context_id]

                    expr_model_type = model_cfg.model_type
                    if model_cfg.model_type == 'geneformer':
                        # loader = load_all_data_SL(data_context, self.geneformer_emb_map, self.args.batch_size, add_kg=self.args.add_kg)
                        loader = load_all_data_SL(data_context, self.geneformer_emb_map, self.args.batch_size, add_kg=0)
                    elif model_cfg.model_type == 'transformer':
                        loader = load_all_data_SL(data_context, self.gene_sent_map, self.args.batch_size, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg)

                    model_dir = os.path.join(model_cfg.path, "model", context)

                    # df_all = []
                    for i, model_savepath in enumerate(os.listdir(model_dir)):
                
                        if model_cfg.model_type == 'geneformer':
                            model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                        elif model_cfg.model_type == 'transformer':
                            model = Transformer_Finetuner(config=transformer_config)
                        
                        predict_res, true_label, cancer, primary_gene, partner_gene = SL_test_prim_partner(device=self.args.device, model=model,
                            pretrain_file=os.path.join(model_dir, model_savepath),
                            data_loader=loader,
                            gene2id_map=self.gene2id_map,
                            cancer2id_map=self.cancer2id_map
                        )
                        pred_result = pd.DataFrame(data=np.concatenate((np.array(primary_gene).reshape(-1,1),
                                                                    np.array(partner_gene).reshape(-1,1),
                                                                    np.array(cancer).reshape(-1,1),
                                                                    predict_res.reshape(-1,1),
                                                                    true_label.reshape(-1,1)), axis=1),
                                                                    columns = ["gene1","gene2","cancer","predict","true"])
                        pred_result = pred_result.astype({"gene1":'str',"gene2":'str',"cancer":'str',"predict":'float32',"true":'float32'})

                        rank_pred_result = pred_result
                        if random_simu:
                            rank_pred_result['predict'] = np.random.permutation(len(pred_result))
                        else:
                            if raw_score:
                                rank_pred_result['predict'] = pred_result['predict']
                            else:
                                rank_pred_result['predict'] = pred_result['predict'].rank(method='min', ascending=True)
                        # df_all.append(rank_pred_result)
                        df_cancer_specific[i].append(rank_pred_result)
                    
                    # df_all = pd.concat(df_all)
                    # df_cancer_specific.append(df_all.groupby(['gene1', 'gene2', 'cancer', 'true'], as_index=False)['predict'].mean())
                
                ## here we concat predictions across different cancers to a single dataframe
                for i in df_cancer_specific.keys():
                    df_cancer_specific[i] = pd.concat(df_cancer_specific[i])

                ## mix
                context = "mix"
                model_cfg = model_dirs[context]

                ## set model parameters
                with open(os.path.join(model_cfg.path, 'params.json'), 'r') as f:
                    model_params = json.load(f)
                for arg in vars(self.args):
                    if arg in model_params:
                        setattr(self.args, arg, model_params[arg])

                transformer_config = self.config_transformer(self.args)

                data = np.load(os.path.join(self.config.SAVED_DATA_DIR, "independent_test_data", f"{data_type}.npy"))

                if model_cfg.model_type == 'geneformer':
                    # loader = load_all_data_SL(data, self.geneformer_emb_map, self.args.batch_size, add_kg=self.args.add_kg)
                    loader = load_all_data_SL(data, self.geneformer_emb_map, self.args.batch_size, add_kg=0)
                elif model_cfg.model_type == 'transformer':
                    loader = load_all_data_SL(data, self.gene_sent_map, self.args.batch_size, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg)

                model_dir = model_dir = os.path.join(model_cfg.path, "model")
                
                # df_mix = []
                df_mix = {i:[] for i in range(5)}
                for i, model_savepath in enumerate(os.listdir(model_dir)):
            
                    if model_cfg.model_type == 'geneformer':
                        model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                    elif model_cfg.model_type == 'transformer':
                        model = Transformer_Finetuner(config=transformer_config)
                    
                    predict_res, true_label, cancer, primary_gene, partner_gene = SL_test_prim_partner(device=self.args.device, model=model,
                        pretrain_file=os.path.join(model_dir, model_savepath),
                        data_loader=loader,
                        gene2id_map=self.gene2id_map,
                        cancer2id_map=self.cancer2id_map
                    )
                    pred_result = pd.DataFrame(data=np.concatenate((np.array(primary_gene).reshape(-1,1),
                                                                np.array(partner_gene).reshape(-1,1),
                                                                np.array(cancer).reshape(-1,1),
                                                                predict_res.reshape(-1,1),
                                                                true_label.reshape(-1,1)), axis=1),
                                                                columns = ["gene1","gene2","cancer","predict","true"])
                    pred_result = pred_result.astype({"gene1":'str',"gene2":'str',"cancer":'str',"predict":'float32',"true":'float32'})
                    
                    rank_pred_result = pred_result
                    if random_simu:
                        rank_pred_result['predict'] = np.random.permutation(len(pred_result))
                    else:
                        if raw_score:
                                rank_pred_result['predict'] = pred_result['predict']
                        else:
                                rank_pred_result['predict'] = pred_result['predict'].rank(method='min', ascending=True)
                    df_mix[i]=rank_pred_result

                # df_mix = pd.concat(df_mix)
                # df_mix = df_mix.groupby(['gene1', 'gene2', 'cancer', 'true'], as_index=False)['predict'].mean()

                ## calculate exact matching counts
                # exact_matching_cancer_specific = independent_exact_matching(df_cancer_specific)
                # exact_matching_mix = independent_exact_matching(df_mix)
                # print(data_type, "cancer-specific:", exact_matching_cancer_specific, "mix:",exact_matching_mix)

                if save_raw:
                    ## save the raw score data
                    ## average across folds
                    # preds = [df[['predict']] for df in df_cancer_specific.values()]
                    # preds_mean = pd.concat(preds, axis=1).mean(axis=1)
                    result = df_cancer_specific[0].copy()
                    for i, df in enumerate(list(df_cancer_specific.values())):
                        result[f'predict_{i}'] = df[['predict']]
                    # result['predict'] = preds_mean
                    if random_simu:
                        result.to_csv(os.path.join(output_dir, f"compare_res_rdn_{data_type}_cancer_specific.csv"), index=False)
                    else:
                        result.to_csv(os.path.join(output_dir, f"compare_res_{expr_model_type}_{data_type}_cancer_specific.csv"), index=False)

                    # preds = [df[['predict']] for df in df_mix.values()]
                    # preds_mean = pd.concat(preds, axis=1).mean(axis=1)
                    result = df_mix[0].copy()
                    for i, df in enumerate(list(df_mix.values())):
                        result[f'predict_{i}'] = df[['predict']]
                    # result['predict'] = preds_mean
                    if random_simu:
                        result.to_csv(os.path.join(output_dir, f"compare_res_rdn_{data_type}_mix.csv"), index=False)
                    else: 
                        result.to_csv(os.path.join(output_dir, f"compare_res_{expr_model_type}_{data_type}_mix.csv"), index=False)
                
                else:
                    for i in range(5):
                        exact_matching_cancer_specific = independent_exact_matching(df_cancer_specific[i])
                        exact_matching_mix = independent_exact_matching(df_mix[i])
                        print(data_type, i, "cancer-specific:", exact_matching_cancer_specific, "mix:",exact_matching_mix)
                        data_save.append({'study':data_type, 'model_fold':i, "cancer-specific":exact_matching_cancer_specific, "mix":exact_matching_mix})
        
        if not save_raw:
            data_save = pd.DataFrame(data_save)
            if random_simu:
                data_save.to_csv(os.path.join(output_dir, f"compare_res_rdn.csv"), index=False)
            else:
                data_save.to_csv(os.path.join(output_dir, f"compare_res_{expr_model_type}.csv"), index=False)


    def infer_primpartner(self):

        if self.experiment != "inference":
            raise Exception("Please set inference configs!")

        output_dir = os.path.join(self.config.EXPERIMENT_DIR, self.experiment, self.config.task.name)
        create_dir(output_dir)

        data_fps = self.config.task.data
        model_dirs = self.config.task.model

        data_all = {}
        for data_name, data_fp in data_fps.items():
            data_all[data_name] = np.load(data_fp)
        
        for data_name, data in data_all.items():
            avgrank_all_model = []
            for scenario, model_cfg in model_dirs.items():
                
                ## set model parameters
                with open(os.path.join(model_cfg.path, 'params.json'), 'r') as f:
                    model_params = json.load(f)
                for arg in vars(self.args):
                    if arg in model_params:
                        setattr(self.args, arg, model_params[arg])
                
                transformer_config = self.config_transformer(self.args)

                if model_cfg.model_type == 'geneformer':
                    loader = load_all_data_SL(data, self.geneformer_emb_map, self.args.batch_size, add_kg=self.args.add_kg)
                elif model_cfg.model_type == 'transformer':
                    loader = load_all_data_SL(data, self.gene_sent_map, self.args.batch_size, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg)
                
                df_all = []
                att_cross_all = []
                att_transformer_all = []
                for model_savename in os.listdir(os.path.join(model_cfg.path, "model")):

                    if model_cfg.model_type == 'geneformer':
                        model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                    elif model_cfg.model_type == 'transformer':
                        model = Transformer_Finetuner(config=transformer_config)
                    
                    predict_res, _, cancer, primary_gene, partner_gene = SL_test_prim_partner(device=self.args.device, model=model,
                        pretrain_file=os.path.join(model_cfg.path, "model", model_savename),
                        data_loader=loader,
                        gene2id_map=self.gene2id_map,
                        cancer2id_map=self.cancer2id_map,
                    )

                    result_df = pd.DataFrame(data=predict_res.reshape(-1,1), columns=["score"])
                    result_df.insert(loc=0, column='cancer', value=cancer)
                    result_df.insert(loc=0, column='partner_gene', value=partner_gene)
                    result_df.insert(loc=0, column='primary_gene', value=primary_gene)
                    # result_df = result_df.sort_values(by=["score"],ascending=False).reset_index()
                    
                    # print(data_name, scenario, result_df[result_df['partner_gene']=='PRKDC'].index[0], '/', str(len(result_df)))
                    df_all.append(result_df)
                    # att_cross_all.append(att_cross)
                    # att_transformer_all.append(att_transformer)

                ## keep all the 5 folds results, remember to comment the line sorting result_df
                merged_df = df_all[0][['primary_gene', 'partner_gene', 'cancer']].copy()
                for i, df in enumerate(df_all):
                    merged_df[f'score_{i+1}'] = df['score']
                merged_df['avg_score'] = merged_df.iloc[:, 3:].mean(axis=1)
                merged_df.to_csv(os.path.join(output_dir, f"{scenario}_{data_name}_5folds.csv"), index=False)

                ## sort by prediction scores
                # df_concat = pd.concat(df_all)
                # avg_score_mean = df_concat.groupby('partner_gene')['score'].mean().reset_index()
                # avg_score_mean = avg_score_mean.sort_values(by=["score"],ascending=False).reset_index()
                # avg_score_mean.to_csv(os.path.join(output_dir, f"{scenario}_{data_name}.csv"), index=False)

                # avg_PRKDC_rank = str(avg_score_mean[avg_score_mean['partner_gene']=='PRKDC'].index[0])+' / '+str(len(avg_score_mean))
                # print("avg", data_name, scenario, avg_PRKDC_rank)
                # avg_PARP1_rank = str(avg_score_mean[avg_score_mean['partner_gene']=='PARP1'].index[0])+' / '+str(len(avg_score_mean))
                # avg_BCL2L1_rank = str(avg_score_mean[avg_score_mean['partner_gene']=='BCL2L1'].index[0])+' / '+str(len(avg_score_mean))
                # avg_NUDT1_rank = str(avg_score_mean[avg_score_mean['partner_gene']=='NUDT1'].index[0])+' / '+str(len(avg_score_mean))

                # avgrank_all_model.append({
                #     "PRKDC": avg_PRKDC_rank,
                #     "PARP1": avg_PARP1_rank,
                #     "BCL2L1": avg_BCL2L1_rank,
                #     "NUDT1": avg_NUDT1_rank,
                # })

                # avg_score_median = df_concat.groupby('partner_gene')['score'].median().reset_index()
                # avg_score_median = avg_score_median.sort_values(by=["score"],ascending=False).reset_index()
                # print("median", data_name, scenario, avg_score_median[avg_score_median['partner_gene']=='PRKDC'].index[0], '/', str(len(avg_score_median)))

                # print("==========================")
                # att_mean = np.mean(att_all, axis=0)
                # with open(os.path.join(output_dir, f"{scenario}_{data_name}_crossatt.pkl"), 'wb') as f:
                #     pkl.dump(att_cross_all, f)
                # with open(os.path.join(output_dir, f"{scenario}_{data_name}_transformeratt.pkl"), 'wb') as f:
                #     pkl.dump(att_transformer_all, f)
            
            # avgrank_all_model = pd.DataFrame(avgrank_all_model)
            # avgrank_all_model.to_csv(os.path.join(output_dir, f"{data_name}_avg_rank.csv"), index=False)


    def get_att(self):

        output_dir = os.path.join(self.config.EXPERIMENT_DIR, self.experiment, self.config.task.name)
        create_dir(output_dir)

        model_dirs = self.config.task.model
        partner_gene = self.config.task.partner_gene
        cancer_id = self.cancer2id_map[self.config.task.cancer]
        # data_all = np.array([[self.gene2id_map['IDH1'], self.gene2id_map[partner_gene], 1, cancer_id]])
        data_all = np.array([[self.gene2id_map[self.config.task.primary_gene], self.gene2id_map[partner_gene], 1, cancer_id]])

        device=self.args.device

        for scenario, model_cfg in model_dirs.items():
            
            print(scenario)
            with open(os.path.join(model_cfg, 'params.json'), 'r') as f:
                model_params = json.load(f)
            for arg in vars(self.args):
                if arg in model_params:
                    setattr(self.args, arg, model_params[arg])
            transformer_config = self.config_transformer(self.args)
            loader = load_all_data_SL(data_all, self.gene_sent_map, 1, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg)
            
            att_cross_all = [[],[],[],[]]
            att_transformer_all = [[],[]]
            for model_savename in os.listdir(os.path.join(model_cfg, "model")):
                pretrain_file=os.path.join(model_cfg, "model", model_savename)
                model = Transformer_Finetuner(config=transformer_config)
                params_pretrain = torch.load(pretrain_file)
                model.load_state_dict(params_pretrain, strict=True)
                for p in model.parameters():
                    p.requires_grad = False
                model = model.to(device)
                model.eval()

                for i, data in enumerate(loader):
                    sent1, mask1, sent2, mask2, label, g1, g2, cancer = data
                    sent1_cuda = sent1.to(device)
                    sent2_cuda = sent2.to(device)
                    mask1_cuda = mask1.to(device)
                    mask2_cuda = mask2.to(device)
                    cross_att_all, transformer_att_all = model.output_att(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda)
                    for j, att_part in enumerate(cross_att_all):
                        att_cross_all[j].append(att_part.detach().cpu())
                    for j, att_part in enumerate(transformer_att_all):
                        att_transformer_all[j].append(att_part.detach().cpu())
            
            with open(os.path.join(output_dir, f"{scenario}_crossatt.pkl"), 'wb') as f:
                pkl.dump(att_cross_all, f)
            with open(os.path.join(output_dir, f"{scenario}_transformeratt.pkl"), 'wb') as f:
                pkl.dump(att_transformer_all, f)



    def permut_primpartner(self, n_sample, n_iter, model_path, savename, model_type='transformer', cancer='Glioma'):
        '''
        n_sample: how many background genes are ranked with each candidate gene together, i.e. the precision of the final averaged ranks
        n_iter: how many different lists of background genes are used for each candidate gene
        '''

        output_dir = os.path.join(self.config.EXPERIMENT_DIR, "inference", self.config.task.type, self.config.task.name)
        create_dir(output_dir)

        prim_gene = self.config.task.primary_gene
        cancer_id = self.cancer2id_map[cancer]
        ## load gene list
        genelist_path = self.config.task.gene_list
        with open(genelist_path) as f:
            genelist = [line.rstrip('\n') for line in f]
        ## add some external genes for reactome DDR gene list
        if "dna" in genelist_path:
            genelist.append("BCL2L1"); genelist.append("NUDT1")
        genes_filt = filt_SL_test_names(genelist, self.gene2id_map, self.gene_sent_map, self.geneformer_emb_map, context=cancer_id)
        print(f"#genes left: {len(genes_filt)}")
        assert len(genelist)>=1

        res_all = []
        for i, partner_candidate in enumerate(genes_filt):
            res_partner = {'gene': partner_candidate}
            for iter in range(1, n_iter+1):
                print(i*(n_iter+1)+iter)
                ## sample data for each iteration
                # data_iter = IDH1_permute_data(self.common_data, partner_candidate, n_sample, seed=i, cancer=cancer)
                data_iter = single_permute_data(self.common_data, prim_gene, partner_candidate, n_sample, seed=i*(n_iter+1)+iter, cancer=cancer)
                if len(data_iter) > 512:    ## maximumly take 512 pairs
                    data_iter = data_iter[:512,:]

                ## set model parameters
                with open(os.path.join(model_path, 'params.json'), 'r') as f:
                    model_params = json.load(f)
                for arg in vars(self.args):
                    if arg in model_params:
                        setattr(self.args, arg, model_params[arg])
                
                transformer_config = self.config_transformer(self.args)

                if model_type == 'geneformer':
                    loader = load_all_data_SL(data_iter, self.geneformer_emb_map, self.args.batch_size, add_kg=self.args.add_kg)
                elif model_type == 'transformer':
                    loader = load_all_data_SL(data_iter, self.gene_sent_map, self.args.batch_size, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg)

                for model_idx, model_savename in enumerate(os.listdir(os.path.join(model_path, "model"))):

                    if model_type == 'geneformer':
                        model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                    elif model_type == 'transformer':
                        model = Transformer_Finetuner(config=transformer_config)
                    
                    predict_res, _, _, primary_gene, partner_gene = SL_test_prim_partner(device=self.args.device, model=model,
                        pretrain_file=os.path.join(model_path, "model", model_savename),
                        data_loader=loader,
                        gene2id_map=self.gene2id_map,
                        cancer2id_map=self.cancer2id_map,
                    )

                    result_df = pd.DataFrame(data=predict_res.reshape(-1,1), columns=["score"])
                    result_df.insert(loc=0, column='partner_gene', value=partner_gene)
                    result_df.insert(loc=0, column='primary_gene', value=primary_gene)
                    result_df = result_df.sort_values(by=["score"],ascending=False).reset_index()

                    rank = result_df[result_df['partner_gene']==partner_candidate].index[0]/len(result_df)
                    # res_partner[f"i_{model_idx}"] = rank
                    # res_partner[f"i_{(model_idx)*(n_iter+1)+iter}"] = rank
                    res_partner[f"i_{model_idx}_{iter}"] = rank

            res_all.append(res_partner)
        
        res_all = pd.DataFrame(res_all)
        # res_all['avg_rank'] = res_all[[f"i_{idx}" for idx in range(5)]].mean(axis=1)
        res_all['avg_rank'] = res_all.iloc[:,1:].mean(axis=1)

        res_all.to_csv(os.path.join(output_dir, f"{savename}_n{n_sample}_iter{n_iter}.csv"), index=False)


    def infer_all(self, output_att=False, output_emb=True):

        output_dir = os.path.join(self.config.EXPERIMENT_DIR, self.experiment, self.config.task.name)
        create_dir(output_dir)

        model_dirs = self.config.task.model
        # data = np.load(self.config.task.data)

        for scenario, model_cfg in model_dirs.items():

            ## set model parameters
            with open(os.path.join(model_cfg.path, 'params.json'), 'r') as f:
                model_params = json.load(f)
            for arg in vars(self.args):
                if arg in model_params:
                    setattr(self.args, arg, model_params[arg])
            if model_cfg.model_type == 'transformer':
                transformer_config = self.config_transformer(self.args)

            att_cross_all = []
            att_transformer_all = []
            emb_cross_all = []
            emb_transformer_all = []

            for model_savename in os.listdir(os.path.join(model_cfg.path, "model")):

                cv = int(model_savename.split("cv")[1][0])
                data = np.load(os.path.join(self.config.task.data, f"test_all_fold_{cv}.npy"))

                if model_cfg.model_type == 'geneformer':
                    loader = load_all_data_SL(data, self.geneformer_emb_map, self.args.batch_size, add_kg=self.args.add_kg)
                elif model_cfg.model_type == 'transformer':
                    loader = load_all_data_SL(data, self.gene_sent_map, self.args.batch_size, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg)

                if model_cfg.model_type == 'geneformer':
                    model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                elif model_cfg.model_type == 'transformer':
                    model = Transformer_Finetuner(config=transformer_config)
                
                out = SL_test_prim_partner(device=self.args.device, model=model,
                    pretrain_file=os.path.join(model_cfg.path, "model", model_savename),
                    data_loader=loader,
                    gene2id_map=self.gene2id_map,
                    cancer2id_map=self.cancer2id_map,
                    output_att=output_att,
                    output_emb=output_emb,
                )

                if output_att and output_emb:
                    predict_res, true_label, att_emb, context, primary_gene, partner_gene = out
                    att_cross, att_transformer, emb_cross, emb_transformer = att_emb
                elif output_att:
                    predict_res, true_label, att_emb, context, primary_gene, partner_gene = out
                    att_cross, att_transformer = att_emb
                elif output_emb:
                    predict_res, true_label, att_emb, context, primary_gene, partner_gene = out
                    emb_cross, emb_transformer = att_emb
                else:
                    predict_res, true_label, context, primary_gene, partner_gene = out

                result_df = pd.DataFrame(data=predict_res.reshape(-1,1), columns=["score"])
                result_df.insert(loc=0, column='cancer', value=context)
                result_df.insert(loc=0, column='partner_gene', value=partner_gene)
                result_df.insert(loc=0, column='primary_gene', value=primary_gene)
                result_df.to_csv(os.path.join(output_dir, f"pred_{scenario}_cv{cv}.csv"), index=False)

                if output_att:
                    att_cross_all.append(att_cross)
                    att_transformer_all.append(att_transformer)
                if output_emb:
                    emb_cross_all.append(emb_cross)
                    emb_transformer_all.append(emb_transformer)        
            
            if output_att:
                with open(os.path.join(output_dir, f"{scenario}_crossatt.pkl"), 'wb') as f:
                    pkl.dump(att_cross_all, f)
                with open(os.path.join(output_dir, f"{scenario}_transformeratt.pkl"), 'wb') as f:
                    pkl.dump(att_transformer_all, f)
            if output_emb:
                with open(os.path.join(output_dir, f"{scenario}_crossemb.pkl"), 'wb') as f:
                    pkl.dump(emb_cross_all, f)
                with open(os.path.join(output_dir, f"{scenario}_transformeremb.pkl"), 'wb') as f:
                    pkl.dump(emb_transformer_all, f)
        

    def get_emb(self):

        output_dir = os.path.join(self.config.EXPERIMENT_DIR, self.experiment, self.config.task.name, self.config.task.cancer)
        create_dir(output_dir)

        model_dirs = self.config.task.model
        # data = np.load(self.config.task.data)
        partner_gene = self.config.task.partner_gene
        cancer_id = self.cancer2id_map[self.config.task.cancer]
        data_all = np.array([[self.gene2id_map[self.config.task.primary_gene], self.gene2id_map[partner_gene], 1, cancer_id]])

        device=self.args.device

        for scenario, model_cfg in model_dirs.items():
            
            print(scenario)
            ## set model parameters
            with open(os.path.join(model_cfg, 'params.json'), 'r') as f:
                model_params = json.load(f)
            for arg in vars(self.args):
                if arg in model_params:
                    setattr(self.args, arg, model_params[arg])
            transformer_config = self.config_transformer(self.args)
            loader = load_all_data_SL(data_all, self.gene_sent_map, 1, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg)

            emb_cross_all = []
            emb_transformer_all = []

            for model_savename in os.listdir(os.path.join(model_cfg, "model")):

                pretrain_file=os.path.join(model_cfg, "model", model_savename)
                model = Transformer_Finetuner(config=transformer_config)
                params_pretrain = torch.load(pretrain_file)
                model.load_state_dict(params_pretrain, strict=True)
                for p in model.parameters():
                    p.requires_grad = False
                model = model.to(device)
                model.eval()

                for i, data in enumerate(loader):
                    sent1, mask1, sent2, mask2, label, g1, g2, cancer = data
                    sent1_cuda = sent1.to(device)
                    sent2_cuda = sent2.to(device)
                    mask1_cuda = mask1.to(device)
                    mask2_cuda = mask2.to(device)

                    h_total, fusion_output = model.output_emb(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda)
                    # h_total[0], h_total[1]
                    if len(fusion_output[0]) > 0:
                        emb_cross_all.append(torch.cat(fusion_output).detach().cpu().numpy())
                    emb_transformer_all.append(torch.cat(h_total, dim=1).detach().cpu().numpy())

            print(len(emb_cross_all))
            print(len(emb_transformer_all))
            with open(os.path.join(output_dir, f"{scenario}_crossemb.pkl"), 'wb') as f:
                pkl.dump(emb_cross_all, f)
            with open(os.path.join(output_dir, f"{scenario}_transformeremb.pkl"), 'wb') as f:
                pkl.dump(emb_transformer_all, f)
                

def SL_test_prim_partner(device, model, pretrain_file, data_loader, gene2id_map, cancer2id_map, output_att=False, output_emb=False):

    id2gene_map = {i:g for g,i in gene2id_map.items()}
    id2cancer_map = {i:c for c,i in cancer2id_map.items()}

    params_pretrain = torch.load(pretrain_file)
    model.load_state_dict(params_pretrain, strict=True)
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    model.eval()

    predict_res = []
    true_label = []
    context = []
    partner_gene_name = []
    primary_gene_name = []
    att_cross = [[],[],[],[]]
    att_transformer = [[],[]]
    emb_cross = [[],[]]
    emb_transformer = [[],[]]

    for i, data in enumerate(data_loader):

        if len(data)==5:
            total_emb, label, g1, g2, cancer = data
            total_emb_cuda = total_emb.to(device).to(torch.float32)
            res = model(total_emb_cuda)
        elif len(data)==8:
            sent1, mask1, sent2, mask2, label, g1, g2, cancer = data
            sent1_cuda = sent1.to(device)
            sent2_cuda = sent2.to(device)
            mask1_cuda = mask1.to(device)
            mask2_cuda = mask2.to(device)
            res = model(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda)

            if output_att:
                cross_att_all, transformer_att_all = model.output_att(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda)
                for j, att_part in enumerate(cross_att_all):
                    att_cross[j].append(att_part.detach().cpu())
                for j, att_part in enumerate(transformer_att_all):
                    att_transformer[j].append(att_part.detach().cpu())
            if output_emb:
                h_total, fusion_output = model.output_emb(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda)
                for k, emb_part in enumerate(fusion_output):
                    emb_cross[k].append(emb_part.detach().cpu())
                for k, emb_part in enumerate(h_total):
                    emb_transformer[k].append(emb_part.transpose(0,1).detach().cpu())


        m = torch.nn.Sigmoid()
        res = torch.squeeze(m(res))
        # res = torch.squeeze(res)
        res = res.detach().cpu()

        predict_res.append(res)
        true_label.append(label)
        context.extend([id2cancer_map[c.item()] for c in cancer])
        partner_gene_name.extend([id2gene_map[g.item()] for g in g2])
        primary_gene_name.extend([id2gene_map[g.item()] for g in g1])
        
    predict_res = torch.cat(predict_res, dim=0)
    true_label = torch.cat(true_label, dim=0)
    if output_att:
        att_cross_out = []
        att_transformer_out = []
        for att_part in att_cross:
            att_cross_out.append(torch.cat(att_part, dim=0).numpy())
        for att_part in att_transformer:
            att_transformer_out.append(torch.cat(att_part, dim=0).numpy())
    if output_emb:
        emb_cross_out = []
        emb_transformer_out = []
        if len(emb_cross[0]) > 0:
            for emb_part in emb_cross:
                emb_cross_out.append(torch.cat(emb_part, dim=0).numpy())
        for emb_part in emb_transformer:
            emb_transformer_out.append(torch.cat(emb_part, dim=0).numpy())

    if not output_att and not output_emb:
        return predict_res.numpy(), true_label.numpy(), context, primary_gene_name, partner_gene_name
    elif output_att:
        return predict_res.numpy(), true_label.numpy(), [att_cross_out, att_transformer_out], context, primary_gene_name, partner_gene_name
    elif output_emb:
        return predict_res.numpy(), true_label.numpy(), [emb_cross_out, emb_transformer_out], context, primary_gene_name, partner_gene_name
    else:
        return predict_res.numpy(), true_label.numpy(), [att_cross_out, att_transformer_out, emb_cross_out, emb_transformer_out], context, primary_gene_name, partner_gene_name



def independent_evaluate(data, type):

    predict = data["predict"].values
    int_predict = np.around(predict, 0)
    true = data["true"].values

    # predict_pair_ranked = list(data.sort_values(by='predict', ascending=False).index)
    # true_pair_ranked = list(data.sort_values(by='true', ascending=True).index)

    topk_range = [10, 20, 30, 50, 100]
    topk = [k for k in topk_range if k < len(data)]

    if type == "binary":
        result = {
            # "auc": metrics.roc_auc_score(true, predict),
            # "aupr": metrics.average_precision_score(true, predict),
            # "f1": metrics.f1_score(true, int_predict),
            # "precision": metrics.precision_score(true, int_predict),
            # "recall": metrics.recall_score(true, int_predict),
            # "acc": metrics.accuracy_score(true, int_predict),
        }
        # ndcg for binary
        # true_hit = list(data[data["true"]==1].index)
        for k in topk:
            # result["ndcg_bin@"+str(k)], result["hit@"+str(k)] = ndcg_bin(k, predict_pair_ranked, true_hit)
            result["ndcg_bin@"+str(k)] = ndcg_score(true[np.newaxis,:], predict[np.newaxis,:], k=k)
            result["precision@"+str(k)] = precision_at_k(true, predict, k=k)
            result["recall@"+str(k)] = recall_at_k(true, predict, k=k)
            result["hit@"+str(k)] = hit_at_k_bin(true, predict, k=k)

    elif type == "rank":
        result = {}
        rel = np.max(true)-true
        for k in topk:
            # result["ndcg@"+str(k)], result["hit@"+str(k)] = ndcg(k, predict_pair_ranked, true_pair_ranked)
            result["ndcg@"+str(k)] = ndcg_score(rel[np.newaxis,:], predict[np.newaxis,:], k=k)
            # result["hit@"+str(k)] = hit_at_k(rel, predict, k=k)
    

    return pd.DataFrame(result, index=[0])


def independent_exact_matching(data):

    data['true_inverse'] = np.max(data['true'].values)-data['true'].values

    grouped = data.groupby(['gene1', 'gene2'])
    match_cnt = 0
    total_cnt = 0

    for (g1, g2), group in grouped:
        if len(group) > 1:
            total_cnt += 1
            predict_order = group.sort_values(by='predict', ascending=False)['cancer'].tolist()
            true_order = group.sort_values(by='true', ascending=False)['cancer'].tolist()
            if predict_order == true_order:
                match_cnt += 1
    
    # return match_cnt / len(grouped)
    return match_cnt / total_cnt