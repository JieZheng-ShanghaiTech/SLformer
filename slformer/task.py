import numpy as np
from sklearn.metrics import ndcg_score
import pandas as pd
import os
import torch
import time
import json
import logging
import wandb
from datasets import load_from_disk

from util import create_dir, calc_pos_weight, ndcg, ndcg_bin, mean_metrics, average_metrics, clear_result, precision_at_k, recall_at_k, hit_at_k, hit_at_k_bin
from train import train, pretrain
from model import MLP, Transformer_Finetuner, Transformer_Pretrain
from dataloader import load_train_data_SL, load_all_data_SL, load_pretrain_data, load_pretrain_data_all




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
            "num_layers": args.num_layers,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "mlp_output_dim": args.mlp_output_dim,
            "add_att": args.add_att,
            "att_nhead": args.att_nhead,
            "random_init": args.random_init,
            
            # "freeze_transformer_encoder": False,
        }

        return transformer_config


    def load_pretrain_checkpoint(self, args, config, cv, model_savename="model.pth"):

        transformer_args = ['n', 'd_model', 'n_head', 'dropout', 'transformer_hidden_dim', 'num_layers',' random_init']

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
            pretrain_checkpoint = os.path.join(model_dir, "model", model_savename)
            params_pretrain = torch.load(pretrain_checkpoint)
            new_state_dict = model.state_dict()
            filt_state_dict = {k: v for k, v in params_pretrain.items() if 'predictor' not in k}
            new_state_dict.update(filt_state_dict)

            return args, new_state_dict

    
    def gsent_pretrain(self, save_model=True, save_result=True):

        criterion = torch.nn.CrossEntropyLoss()

        random_init = self.args.random_init

        self.args.mlp_output_dim = len(self.gene2id_map)
        # self.args.mlp_output_dim = 9    #num of cancer types
        # gene2anno_map = self.common_data["gene2go_map"]
        # self.args.mlp_output_dim = len(set(gene2anno_map.values()))+1
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

        # model_save_path = os.path.join(model_root_dir, f"model.pth")
        # train_loader, test_loader = load_pretrain_data(data_train, data_test, batch_size=self.args.batch_size, emb_mtx=self.geneformer_emb_mtx, n=self.args.n, gene2anno_map=gene2anno_map, random_init=random_init)
        train_loader, test_loader = load_pretrain_data(data_train, data_test, batch_size=self.args.batch_size, emb_mtx=self.geneformer_emb_mtx, n=self.args.n, gene2anno_map=None, random_init=random_init)
        # data_loader= load_pretrain_data_all(filt_data, batch_size=self.args.batch_size, emb_mtx=self.geneformer_emb_mtx, n=self.args.n, gene2anno_map=None, random_init=random_init)
        # data_loader= load_pretrain_data_all(filt_data, batch_size=self.args.batch_size, emb_mtx=self.geneformer_emb_mtx, n=self.args.n, gene2anno_map=gene2anno_map, random_init=random_init)
        model = Transformer_Pretrain(config=transformer_config)

        pretrain(self.args.device, model, criterion, self.args, train_loader, test_loader, model_root_dir, result_path, save_model=save_model, save_result=save_result)
        # pretrain(self.args.device, model, criterion, self.args, data_loader, model_root_dir, result_path, save_model=save_model, save_result=save_result)


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

                    # pos_weight = calc_pos_weight(data_train)
                    # pos_weight = torch.tensor([pos_weight]).to(device=torch.device("cuda:" + str(self.args.device)))
                    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    # print("pos/neg weight=", pos_weight)

                    logging.info(f"{cancer_type}_cv{cv}, train data size={len(data_train)}, test data size={len(data_test)}")
                    
                    model_save_path = os.path.join(model_root_dir, cancer_type, f"model_{cancer_type}_cv{cv}.pth")
                    create_dir(os.path.join(model_root_dir, cancer_type))

                    if model_class == 'geneformer':
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.geneformer_emb_map, self.args.batch_size)
                        model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                    elif model_class == 'transformer':
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.gene_sent_map, self.args.batch_size, self.args.n, self.args.anchor, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, augmentation=self.args.augmentation)
                        if 'mix_checkpoint' in self.config.task or 'pretrain_checkpoint' in self.config.task:
                            if 'pretrain_checkpoint' in self.config.task:
                                ckp_args, ckp = self.load_pretrain_checkpoint(self.args, self.config, cv, model_savename=self.config.task.pretrain_checkpoint.save_name)
                            else:
                                ckp_args, ckp = self.load_pretrain_checkpoint(self.args, self.config, cv)
                            transformer_config = self.config_transformer(ckp_args)
                            model = Transformer_Finetuner(config=transformer_config)
                            model.load_state_dict(ckp, strict=True)
                        else:
                            transformer_config = self.config_transformer(self.args)
                            model = Transformer_Finetuner(config=transformer_config)  
                        
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.gene_sent_map, self.args.batch_size, self.args.n, self.args.anchor, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, augmentation=self.args.augmentation)                      

                    train(self.args.device, model, criterion, m, self.args, train_loader, model_save_path, result_path, test_loader, save_model=save_model, save_result=save_result, model_class=model_class, wandb_run=run)

                # get average results
                if wandb_track:
                    avg_metrics = mean_metrics(result_path)
                    run.log(avg_metrics)

                if save_result:
                    average_metrics(result_path)

            if wandb_track: 
                run.finish()
        


        if self.experiment == 'mix' or self.experiment == 'mix_all':

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
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.geneformer_emb_map, self.args.batch_size)
                        model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                    elif model_class == 'transformer':
                        train_loader, test_loader = load_train_data_SL(data_test, data_train, self.gene_sent_map, self.args.batch_size, self.args.n, self.args.anchor, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, augmentation=self.args.augmentation)
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
                    train_loader, test_loader = load_train_data_SL(data_test, data_train, self.geneformer_emb_map, self.args.batch_size)
                    model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                elif model_class == 'transformer':
                    train_loader, test_loader = load_train_data_SL(data_test, data_train, self.gene_sent_map, self.args.batch_size, self.args.n, self.args.anchor, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, augmentation=self.args.augmentation)
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
            

    def independent_test(self):

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
                    loader = load_all_data_SL(data, self.geneformer_emb_map, self.args.batch_size)
                elif model_cfg.model_type == 'transformer':
                    loader = load_all_data_SL(data, self.gene_sent_map, self.args.batch_size, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx)

                df_all = []
                if context == 'mix':
                    model_dir = os.path.join(model_cfg.path, "model")
                else:
                    model_dir = os.path.join(model_cfg.path, "model", context)

                for i, model_savepath in enumerate(os.listdir(model_dir)):
            
                    if model_cfg.model_type == 'geneformer':
                        model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                    elif model_cfg.model_type == 'transformer':
                        model = Transformer_Finetuner(config=self.transformer_config)
                    
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

            for scenario, model_cfg in model_dirs.items():
                
                ## set model parameters
                with open(os.path.join(model_cfg.path, 'params.json'), 'r') as f:
                    model_params = json.load(f)
                for arg in vars(self.args):
                    if arg in model_params:
                        setattr(self.args, arg, model_params[arg])
                
                transformer_config = self.config_transformer(self.args)

                if model_cfg.model_type == 'geneformer':
                    loader = load_all_data_SL(data, self.geneformer_emb_map, self.args.batch_size)
                elif model_cfg.model_type == 'transformer':
                    loader = load_all_data_SL(data, self.gene_sent_map, self.args.batch_size, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx)
                
                df_all = []
                for model_savename in os.listdir(os.path.join(model_cfg.path, "model")):

                    if model_cfg.model_type == 'geneformer':
                        model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                    elif model_cfg.model_type == 'transformer':
                        model = Transformer_Finetuner(config=transformer_config)
                    
                    predict_res, _, _, primary_gene, partner_gene = SL_test_prim_partner(device=self.args.device, model=model,
                        pretrain_file=os.path.join(model_cfg.path, "model", model_savename),
                        data_loader=loader,
                        gene2id_map=self.gene2id_map,
                        cancer2id_map=self.cancer2id_map
                    )

                    result_df = pd.DataFrame(data=predict_res.reshape(-1,1), columns=["score"])
                    result_df.insert(loc=0, column='partner_gene', value=partner_gene)
                    result_df.insert(loc=0, column='primary_gene', value=primary_gene)
                    result_df = result_df.sort_values(by=["score"],ascending=False).reset_index()
                    print(data_name, scenario, result_df[result_df['partner_gene']=='PRKDC'].index[0], '/', str(len(result_df)))
                    df_all.append(result_df)

                df_concat = pd.concat(df_all)
                avg_score = df_concat.groupby('partner_gene')['score'].mean().reset_index()
                avg_score = avg_score.sort_values(by=["score"],ascending=False)
                avg_score.to_csv(os.path.join(output_dir, f"{scenario}_{data_name}.csv"), index=False)




def SL_test_prim_partner(device, model, pretrain_file, data_loader, gene2id_map, cancer2id_map):

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
        context.extend([id2cancer_map[c.item()] for c in cancer])
        partner_gene_name.extend([id2gene_map[g.item()] for g in g2])
        primary_gene_name.extend([id2gene_map[g.item()] for g in g1])
        
    predict_res = torch.cat(predict_res, dim=0)
    true_label = torch.cat(true_label, dim=0)

    return predict_res.numpy(), true_label.numpy(), context, primary_gene_name, partner_gene_name




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
