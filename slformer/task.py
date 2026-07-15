import numpy as np
from sklearn.metrics import ndcg_score
import pandas as pd
import os
import pickle as pkl
import torch
import time
import json
import logging
import wandb

from util import create_dir, mean_metrics, average_metrics, precision_at_k, recall_at_k, hit_at_k, hit_at_k_bin
from train import train
from model import MLP, SLformer
from dataset import load_train_data_SL, load_all_data_SL
from prepare_data import single_permute_data, filt_SL_test_names


EXPERIMENT_REGISTRY = {
    "cancer_specific": "train",
    "mixed_cancer": "train",
    "cross_cancer": "train",

    "independent_test": "inference",
    "IDH1_DDR_inference": "inference",
    "IDH1_PRKDC_inference": "inference",
    "IDH1_permute": "inference",
    "get_emb": "inference",
    "get_att": "inference",
}

def _config_transformer(args):

    transformer_config = {
        "d_model": args.d_model,
        "n_head": args.n_head,
        "dropout": args.dropout,
        "vocab_size": args.vocab_size,
        "transformer_hidden_dim": args.transformer_hidden_dim,
        "transformer_num_layers": args.transformer_num_layers,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_output_dim": args.mlp_output_dim,
        "add_att": args.add_att,
        "att_nhead": args.att_nhead,
        "att_num_layers": args.att_num_layers,
        "random_init": args.random_init,
    }

    return transformer_config


class Training_Models():

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
        self.args.vocab_size = len(self.gene2id_map)


    def run_experiment(self, save_model=False, save_result=True, wandb_track=False):
        
        run, experiment_dir, result_root_dir, model_root_dir = self._setup_experiment(save_model, save_result, wandb_track)

        if 'cancer_specific' in self.experiment:
            self._run_cancer_specific(
                run, result_root_dir, model_root_dir,
                save_model, save_result
            )

        elif 'mix' in self.experiment:
            self._run_mix(
                run, result_root_dir, model_root_dir,
                save_model, save_result
            )

        elif 'cross_cancer' in self.experiment:
            self._run_cross_cancer(
                run, result_root_dir, model_root_dir,
                save_model, save_result
            )

        if wandb_track and run is not None:
            run.finish()
    

    def _setup_experiment(self, save_model, save_result, wandb_track):

        if wandb_track:
            wandb.init()
            for arg in vars(self.args):
                if hasattr(wandb.config, arg):
                    setattr(self.args, arg, getattr(wandb.config, arg))

        self.criterion = torch.nn.BCELoss()
        self.activation = torch.nn.Sigmoid()
        self.model_class = self.config.model_type

        curr_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
        experiment_dir = os.path.join(self.config.EXPERIMENT_DIR, f"{self.experiment}", f"{self.model_class}", curr_time)
        result_root_dir = os.path.join(experiment_dir, "result")
        model_root_dir = os.path.join(experiment_dir, "model")

        create_dir(experiment_dir)
        if save_result:
            create_dir(result_root_dir)
        if save_model:
            create_dir(model_root_dir)

        logging.getLogger().addHandler(
            logging.FileHandler(os.path.join(experiment_dir, "log.txt"), mode='w')
        )

        params = {**vars(self.args), **self.config}
        with open(os.path.join(experiment_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

        run = None
        if wandb_track:
            run = wandb.init(
                group=self.experiment,
                name=f"{self.experiment}_{curr_time}",
                reinit=True
            )

        return run, experiment_dir, result_root_dir, model_root_dir

    def _build_model_dataloader(self, data_train, data_test, cv=None):
        if self.model_class == 'geneformer':
            train_loader, test_loader = load_train_data_SL(
                data_test, data_train,
                self.geneformer_emb_map,
                self.args.batch_size,
                add_kg=self.args.add_kg
            )
            model = MLP(
                num_layers=2,
                input_dim=self.args.mlp_input_dim,
                hidden_dim=self.args.mlp_hidden_dim,
                output_dim=self.args.mlp_output_dim
            )

        elif self.model_class == 'transformer':
            train_loader, test_loader = load_train_data_SL(
                data_test, data_train,
                self.gene_sent_map,
                self.args.batch_size,
                self.args.n,
                bi_rpr=True,
                sent_mask=self.sent_mask_map,
                emb_mtx=self.geneformer_emb_mtx,
                add_kg=self.args.add_kg
            )

            model = SLformer(config=_config_transformer(self.args))

        return model, train_loader, test_loader
    
    def _run_cv_loop(self, cancer_type, result_path, model_dir, save_model, save_result, run):
        
        for cv in range(1, 6):
            data_test = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"test_{cancer_type}_fold_{cv}.npy"))
            data_train = np.load(os.path.join(self.config.SAVED_DATA_DIR, "SL_train_test_data", self.experiment, f"train_{cancer_type}_fold_{cv}.npy"))

            model, train_loader, test_loader = self._build_model_dataloader(data_train, data_test, cv)

            model_path = os.path.join(model_dir, f"model_{cancer_type}_cv{cv}.pth")
            create_dir(os.path.dirname(model_path))

            train(
                self.args.device,
                model,
                self.criterion,
                self.activation,
                self.args,
                train_loader,
                model_path,
                result_path,
                test_loader,
                save_model=save_model,
                save_result=save_result,
                model_class=self.model_class,
                wandb_run=run
            )
    
    def _run_cancer_specific(self, run, result_root_dir, model_root_dir,
                         save_model, save_result):

        for cancer in self.config.task.cancer:
            result_path = os.path.join(
                result_root_dir, f"train_result_{cancer}.csv"
            )
            self._run_cv_loop(
                cancer,
                result_path,
                os.path.join(model_root_dir, cancer),
                save_model,
                save_result,
                run
            )

            if save_result:
                average_metrics(result_path)
            if run is not None:
                run.log(mean_metrics(result_path))
    
    def _run_mix(self, run, result_root_dir, model_root_dir,
             save_model, save_result):

        for cancer in self.config.task.cancer:
            result_path = os.path.join(
                result_root_dir, f"train_result_{cancer}.csv"
            )
            self._run_cv_loop(
                cancer,
                result_path,
                model_root_dir,
                save_model,
                save_result,
                run
            )

            if save_result:
                average_metrics(result_path)
            if run is not None:
                run.log(mean_metrics(result_path))
    
    def _run_cross_cancer(self, run, result_root_dir, model_root_dir,
                      save_model, save_result):

        result_path = os.path.join(result_root_dir, "train_result_cross_cancer.csv")
        
        ## no cv loop, set up experiment separately
        for cancer in self.config.task.cancer:
            data_train = np.load(os.path.join(self.config.SAVED_DATA_DIR,"SL_train_test_data",self.experiment,f"train_{cancer}.npy"))
            data_test = np.load(os.path.join(self.config.SAVED_DATA_DIR,"SL_train_test_data",self.experiment,f"test_{cancer}.npy"))

            model, train_loader, test_loader = self._build_model_dataloader(data_train, data_test)

            model_path = os.path.join(model_root_dir, f"model_transfer2{cancer}.pth")

            train(
                self.args.device,
                model,
                self.criterion,
                self.activation,
                self.args,
                train_loader,
                model_path,
                result_path,
                test_loader,
                save_model=save_model,
                save_result=save_result,
                model_class=self.model_class,
                wandb_run=run
            )

        if save_result:
            average_metrics(result_path)
        if run is not None:
            run.log(mean_metrics(result_path))



class Inference():

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
        self.args.vocab_size = len(self.gene2id_map)

        self.cancer_list = common_data["cancer_list"]
        self.cancer2id_map = common_data["cancer2id_map"]
        self.id2cancer_map = {i:c for i,c in enumerate(self.cancer_list)}
    

    def run_experiment(self, experiment_type):

        if experiment_type == "independent_test":
            if self.experiment == "independent_test":
                self.independent_test()
            elif self.experiment == "independent_test_on_mix":
                self.independent_test_on_mix()
        elif experiment_type == "IDH1_DDR_inference":
            self.infer_primpartner()
        elif experiment_type == "IDH1_PRKDC_inference":
            self.infer_primpartner()
        elif experiment_type == "IDH1_permute":
            cancer = self.config.task.cancer
            for name, model_path in self.config.model.items():
                self.permut_primpartner(n_sample=1000, n_iter=10,
                                        model_path=model_path, savename=name, 
                                        model_type='transformer', cancer=cancer)

        elif experiment_type == "get_emb":
            self.get_emb()
        elif experiment_type == "get_att":
            self.get_att()

    
    def _load_model_params(self, model_cfg):
        model_path = model_cfg if isinstance(model_cfg, str) else model_cfg.path  
        with open(os.path.join(model_path, "params.json"), "r") as f:
            model_params = json.load(f)
        for arg in vars(self.args):
            if arg in model_params:
                setattr(self.args, arg, model_params[arg])


    def _build_loader(self, data, model_type):
        if model_type == "geneformer":
            return load_all_data_SL(
                data,
                self.geneformer_emb_map,
                self.args.batch_size,
                add_kg=self.args.add_kg
            )
        elif model_type == "transformer":
            return load_all_data_SL(
                data,
                self.gene_sent_map,
                self.args.batch_size,
                self.args.n,
                bi_rpr=True,
                sent_mask=self.sent_mask_map,
                emb_mtx=self.geneformer_emb_mtx,
                add_kg=self.args.add_kg
            )
    
    def _build_model(self, model_type):
        if model_type == "geneformer":
            return MLP(
                num_layers=2,
                input_dim=self.args.mlp_input_dim,
                hidden_dim=self.args.mlp_hidden_dim,
                output_dim=self.args.mlp_output_dim
            )
        elif model_type == "transformer":
            config = _config_transformer(self.args)
            return SLformer(config=config)
    

    def _run_models_on_loader(self, model_dir, model_cfg, loader, output_att=False, output_emb=False):
        
        results = []

        # model_dir = os.path.join(model_cfg.path, "model")
        for model_ckpt in os.listdir(model_dir):

            model = self._build_model(model_cfg.model_type)

            out = SL_test_prim_partner(
                device=self.args.device,
                model=model,
                pretrain_file=os.path.join(model_dir, model_ckpt),
                data_loader=loader,
                gene2id_map=self.gene2id_map,
                cancer2id_map=self.cancer2id_map,
                output_att=output_att,
                output_emb=output_emb
            )
            results.append(out)

        return results


    def independent_test(self):

        output_dir = os.path.join(self.config.EXPERIMENT_DIR, self.experiment)
        pred_dir = os.path.join(output_dir, "pred")
        eval_dir = os.path.join(output_dir, "eval")
        eval_cv_dir = os.path.join(output_dir, "eval_cv")

        test_datasets = self.config.SL_dataset
        model_dirs = self.config.task.model

        for data_type, meta in test_datasets.items():
            if data_type == "general":
                continue

            context = meta["context"]
            model_cfg = model_dirs[context]

            self._load_model_params(model_cfg)

            data = np.load(
                os.path.join(
                    self.config.SAVED_DATA_DIR,
                    "independent_test_data",
                    f"{data_type}.npy"
                )
            )

            loader = self._build_loader(data, model_cfg.model_type)
            if context == 'mix':
                model_dir = os.path.join(model_cfg.path, "model")
            else:
                model_dir = os.path.join(model_cfg.path, "model", context)
            
            create_dir(os.path.join(pred_dir, context))
            create_dir(os.path.join(eval_dir, context))
            create_dir(os.path.join(eval_cv_dir, context))

            ## do inference
            fold_results = self._run_models_on_loader(model_dir, model_cfg, loader)

            eval_all = []
            for i, (pred, true, cancer, g1, g2) in enumerate(fold_results):

                df = pd.DataFrame({
                    "gene1": g1,
                    "gene2": g2,
                    "cancer": cancer,
                    "predict": pred,
                    "true": true
                })

                df.to_csv(
                    os.path.join(pred_dir, context, f"{data_type}_{i}.csv"),
                    index=False
                )

                eval_df = independent_evaluate(df, meta.label_type)
                eval_all.append(eval_df)

            eval_concat = pd.concat(eval_all)
            mean = eval_concat.mean()
            std = eval_concat.std()

            summary = pd.DataFrame([f"{mean[i]:.4f} ({std[i]:.4f})" for i in range(len(mean))]).T
            summary.columns = eval_concat.columns
            
            summary.to_csv(
                os.path.join(eval_dir, context, f"{data_type}_{model_cfg.model_type}.csv"),
                index=False
            )
            eval_concat.to_csv(
                os.path.join(eval_cv_dir, context, f"{data_type}_{model_cfg.model_type}.csv"),
                index=False
            )
    

    def independent_test_on_mix(self, save_raw=False, raw_score=False):

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

        data_save = []

        for data_type in list(test_datasets.keys()):
            ## test on cancer-specific datasets only
            if data_type != "general" and test_datasets[data_type]["context"]=="mix":

                context_all = test_datasets[data_type]["context_all"]
                
                ##  =======cancer-specific models=======
                df_cancer_specific = {i:[] for i in range(5)}
                for context in context_all:

                    model_cfg = model_dirs[context]
                    self._load_model_params(model_cfg)

                    data = np.load(os.path.join(self.config.SAVED_DATA_DIR, "independent_test_data", f"{data_type}.npy"))

                    ## filt context-specific data
                    context_id = self.cancer2id_map[context]
                    data_context = data[data[:, 3] == context_id]

                    expr_model_type = model_cfg.model_type
                    loader = self._build_loader(data_context, expr_model_type)
                    model_dir = os.path.join(model_cfg.path, "model", context)

                    fold_results = self._run_models_on_loader(model_dir, model_cfg, loader)
                    for i, (pred, true, cancer, g1, g2) in enumerate(fold_results):
                        pred_result = pd.DataFrame({
                            "gene1": g1,
                            "gene2": g2,
                            "cancer": cancer,
                            "predict": pred,
                            "true": true
                        })
                        rank_pred_result = pred_result
                        if raw_score:
                            rank_pred_result['predict'] = pred_result['predict']
                        else:
                            rank_pred_result['predict'] = pred_result['predict'].rank(method='min', ascending=True)
                        df_cancer_specific[i].append(rank_pred_result)
                    
                ## here we concat predictions across different cancers to a single dataframe
                for i in df_cancer_specific.keys():
                    df_cancer_specific[i] = pd.concat(df_cancer_specific[i])

                ## =======mix models=======
                context = "mix"
                model_cfg = model_dirs[context]

                self._load_model_params(model_cfg)
                data = np.load(os.path.join(self.config.SAVED_DATA_DIR, "independent_test_data", f"{data_type}.npy"))

                loader = self._build_loader(data, model_cfg.model_type)
                model_dir = os.path.join(model_cfg.path, "model")
                
                df_mix = {i:[] for i in range(5)}
                fold_results = self._run_models_on_loader(model_dir, model_cfg, loader)
                for i, (pred, true, cancer, g1, g2) in enumerate(fold_results):
                    pred_result = pd.DataFrame({
                        "gene1": g1,
                        "gene2": g2,
                        "cancer": cancer,
                        "predict": pred,
                        "true": true
                    })
                    rank_pred_result = pred_result
                    if raw_score:
                        rank_pred_result['predict'] = pred_result['predict']
                    else:
                        rank_pred_result['predict'] = pred_result['predict'].rank(method='min', ascending=True)
                    df_mix[i]=rank_pred_result

                if save_raw:
            
                    result = df_cancer_specific[0].copy()
                    for i, df in enumerate(list(df_cancer_specific.values())):
                        result[f'predict_{i}'] = df[['predict']]
                    
                    result.to_csv(os.path.join(output_dir, f"compare_res_{expr_model_type}_{data_type}_cancer_specific.csv"), index=False)

                    result = df_mix[0].copy()
                    for i, df in enumerate(list(df_mix.values())):
                        result[f'predict_{i}'] = df[['predict']]
                    result.to_csv(os.path.join(output_dir, f"compare_res_{expr_model_type}_{data_type}_mix.csv"), index=False)
                
                else:
                    for i in range(5):
                        exact_matching_cancer_specific = independent_exact_matching(df_cancer_specific[i])
                        exact_matching_mix = independent_exact_matching(df_mix[i])
                        print(data_type, i, "cancer-specific:", exact_matching_cancer_specific, "mix:",exact_matching_mix)
                        data_save.append({'study':data_type, 'model_fold':i, "cancer-specific":exact_matching_cancer_specific, "mix":exact_matching_mix})
        
        if not save_raw:
            data_save = pd.DataFrame(data_save)
            data_save.to_csv(os.path.join(output_dir, f"compare_res_{expr_model_type}.csv"), index=False)
            


    def infer_primpartner(self):

        output_dir = os.path.join(self.config.EXPERIMENT_DIR,self.experiment,self.config.task.name)
        create_dir(output_dir)

        for data_name, data_fp in self.config.task.data.items():
            data = np.load(data_fp)

            for scenario, model_cfg in self.config.task.model.items():

                self._load_model_params(model_cfg)
                loader = self._build_loader(data, model_cfg.model_type)

                model_dir=os.path.join(model_cfg.path, "model")
                results = self._run_models_on_loader(model_dir, model_cfg, loader)

                merged = []
                for i, (pred, _, _, g1, g2) in enumerate(results):
                    df = pd.DataFrame({
                        "primary_gene": g1,
                        "partner_gene": g2,
                        f"score_{i}": pred
                    })
                    merged.append(df)

                out = merged[0][["primary_gene", "partner_gene"]].copy()
                for df in merged:
                    out[df.columns[-1]] = df.iloc[:, -1]

                out["avg_score"] = out.iloc[:, 2:].mean(axis=1)
                out.to_csv(
                    os.path.join(output_dir, f"{scenario}_{data_name}_5folds.csv"),
                    index=False
                )
    

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
                
                transformer_config = _config_transformer(self.args)

                if model_type == 'geneformer':
                    loader = load_all_data_SL(data_iter, self.geneformer_emb_map, self.args.batch_size, add_kg=self.args.add_kg)
                elif model_type == 'transformer':
                    loader = load_all_data_SL(data_iter, self.gene_sent_map, self.args.batch_size, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg)

                for model_idx, model_savename in enumerate(os.listdir(os.path.join(model_path, "model"))):

                    if model_type == 'geneformer':
                        model = MLP(num_layers=2, input_dim=self.args.mlp_input_dim, hidden_dim=self.args.mlp_hidden_dim, output_dim=self.args.mlp_output_dim)
                    elif model_type == 'transformer':
                        model = SLformer(config=transformer_config)
                    
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



    def get_att(self):

        output_dir = os.path.join(self.config.EXPERIMENT_DIR, self.experiment, self.config.task.name)
        create_dir(output_dir)

        model_dirs = self.config.task.model
        partner_gene = self.config.task.partner_gene
        cancer_id = self.cancer2id_map[self.config.task.cancer]
        ## construct a 1-line data
        data_all = np.array([[self.gene2id_map[self.config.task.primary_gene], self.gene2id_map[partner_gene], 1, cancer_id]])

        device=self.args.device

        for scenario, model_cfg in model_dirs.items():
            
            self._load_model_params(model_cfg)
            loader = load_all_data_SL(data_all, self.gene_sent_map, 1, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg)
            
            att_cross_all = [[],[],[],[]]
            att_transformer_all = [[],[]]

            for model_savename in os.listdir(os.path.join(model_cfg, "model")):
                model = self._build_model("transformer")

                pretrain_file=os.path.join(model_cfg, "model", model_savename)
                params_pretrain = torch.load(pretrain_file)
                model.load_state_dict(params_pretrain, strict=True)
                for p in model.parameters():
                    p.requires_grad = False
                model = model.to(device)
                model.eval()

                for i, data in enumerate(loader):
                    sent1, mask1, sent2, mask2, _, _, _, _ = data
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


    def get_emb(self):

        output_dir = os.path.join(self.config.EXPERIMENT_DIR, self.experiment, self.config.task.name, self.config.task.cancer)
        create_dir(output_dir)

        model_dirs = self.config.task.model
        partner_gene = self.config.task.partner_gene
        cancer_id = self.cancer2id_map[self.config.task.cancer]
        data_all = np.array([[self.gene2id_map[self.config.task.primary_gene], self.gene2id_map[partner_gene], 1, cancer_id]])

        device=self.args.device

        for scenario, model_cfg in model_dirs.items():
            
            ## set model parameters
            self._load_model_params(model_cfg)
            loader = load_all_data_SL(data_all, self.gene_sent_map, 1, self.args.n, bi_rpr=True, sent_mask=self.sent_mask_map, emb_mtx=self.geneformer_emb_mtx, add_kg=self.args.add_kg)

            emb_cross_all = []
            emb_transformer_all = []

            for model_savename in os.listdir(os.path.join(model_cfg, "model")):
                model = self._build_model("transformer")

                pretrain_file=os.path.join(model_cfg, "model", model_savename)
                params_pretrain = torch.load(pretrain_file)
                model.load_state_dict(params_pretrain, strict=True)
                for p in model.parameters():
                    p.requires_grad = False
                model = model.to(device)
                model.eval()

                for i, data in enumerate(loader):
                    sent1, mask1, sent2, mask2, _, _, _, _ = data
                    sent1_cuda = sent1.to(device)
                    sent2_cuda = sent2.to(device)
                    mask1_cuda = mask1.to(device)
                    mask2_cuda = mask2.to(device)

                    h_total, fusion_output = model.output_emb(sent1_cuda, mask1_cuda, sent2_cuda, mask2_cuda)
                    # h_total[0], h_total[1]
                    if len(fusion_output[0]) > 0:
                        emb_cross_all.append(torch.cat(fusion_output).detach().cpu().numpy())
                    emb_transformer_all.append(torch.cat(h_total, dim=1).detach().cpu().numpy())

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

    topk_range = [10, 20, 30, 50, 100]
    topk = [k for k in topk_range if k < len(data)]

    if type == "binary":
        result = {}
        for k in topk:
            result["ndcg_bin@"+str(k)] = ndcg_score(true[np.newaxis,:], predict[np.newaxis,:], k=k)
            result["precision@"+str(k)] = precision_at_k(true, predict, k=k)
            result["recall@"+str(k)] = recall_at_k(true, predict, k=k)
            result["hit@"+str(k)] = hit_at_k_bin(true, predict, k=k)

    elif type == "rank":
        result = {}
        rel = np.max(true)-true
        for k in topk:
            result["ndcg@"+str(k)] = ndcg_score(rel[np.newaxis,:], predict[np.newaxis,:], k=k)
            
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