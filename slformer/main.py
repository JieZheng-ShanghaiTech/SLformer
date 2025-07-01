import argparse
import yaml
import easydict
import logging

from util import set_seed
# from preprocess import Data_Preprocess
from task import Validation_Experiment



parser = argparse.ArgumentParser(description='SL prediction')

parser.add_argument('--data_config_file', type=str, default="./config/data_preprocess.yaml",
                    help='data preprocess config file path')
# parser.add_argument('--config_file', type=str, default="./config/mix.yaml",
#                     help='config file path')
parser.add_argument('--config_file', type=str, default="./config/all_inference.yaml",
                    help='config file path')
# parser.add_argument('--config_file', type=str, default="config/IDH1_permute.yaml",
#                     help='config file path')
# parser.add_argument('--config_file', type=str, default="./config/get_att.yaml",
#                     help='config file path')
parser.add_argument('--wandb_track', type=int, default=0,
                    help='whether to track model performance on wandb')
parser.add_argument('--save_model', type=int, default=0,
                    help='whether to save the model checkpoints')
parser.add_argument('--save_result', type=int, default=1,
                    help='whether to save the performance of the models')

parser.add_argument('--n', type=int, default=10,
                    help='genesentence length n')
parser.add_argument('--add_kg', type=int, default=1,
                    help='whether to add KG embedding')
parser.add_argument('--augmentation', type=str, default=None,
                    help='whether to augment the gene sentence input')
parser.add_argument('--anchor', type=int, default=1,
                    help='whether to include the head anchor gene in a gene sentence')

parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--early_stop', type=int, default=5,
                    help="Early stopping patience")
# optimizer
parser.add_argument('--transformer_lr', type=float, default=1e-5,
                    help='learning rate (default: 5e-5)')
parser.add_argument('--predictor_lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.99),
                    help='')
parser.add_argument('--eps', type=float, default=1e-05,
                    help='')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='')
parser.add_argument('--lr_factor', type=float, default=0.5,
                    help='')
parser.add_argument('--lr_patience', type=float, default=3,
                    help='')
# MLP
parser.add_argument('--mlp_input_dim', type=int, default=256*2,
                    help='input dim for MLP')
parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                    help='hidden dim for MLP')
parser.add_argument('--mlp_output_dim', type=int, default=1,
                    help='output dim for MLP')
# Transformer
parser.add_argument('--d_model', type=int, default=256*2,
                    help='')
parser.add_argument('--n_head', type=int, default=1,
                    help='')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='')
parser.add_argument('--transformer_hidden_dim', type=int, default=256,
                    help='')
parser.add_argument('--transformer_num_layers', type=int, default=2,
                    help='')
parser.add_argument('--att_num_layers', type=int, default=1,
                    help='')
parser.add_argument('--add_att', type=int, default=1,
                    help='')
parser.add_argument('--att_nhead', type=int, default=2,
                    help='')
parser.add_argument('--random_init', type=int, default=1,
                    help='')



args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

set_seed(args.random_init)

with open(args.data_config_file, 'r') as f:
    data_config = easydict.EasyDict(yaml.safe_load(f))
with open(args.config_file, 'r') as f:
    config = easydict.EasyDict(yaml.safe_load(f))

#################################################
# Override args with parameters from the config file only for cancer-specific tasks
if config.task.type == "cancer_specific":
    for key, value in config.params.items():
        # Convert specific parameters to the correct types
        if key in ['dropout', 'eps', 'weight_decay', 'transformer_lr', 'predictor_lr']:
            setattr(args, key, float(value))
        elif key in ['batch_size', 'n_head', 'num_layers', 'n', 'early_stop', 'lr_patience', 'device']:
            setattr(args, key, int(value))
        else:
            setattr(args, key, value)
# if you pass a parameter via the command line, it will take precedence over the YAML file 
# command-line arguments have the highest priority
#################################################


if "pretrain" in args.config_file:
    from preprocess_sc import Data_Preprocess
else:
    from preprocess import Data_Preprocess

data_preprocess = Data_Preprocess(data_config)
common_data = data_preprocess.get_common_data(sent_n=200)


experiment_set = Validation_Experiment(
    config=config,
    args = args,
    common_data=common_data,
)


if "pretrain" in args.config_file:

    from preprocess_sc import Data_Preprocess

    experiment_set.gsent_pretrain(
        save_model=args.save_model,
        save_result=args.save_result,
    )


elif "cancer_specific" in args.config_file or "mix" in args.config_file or "cross_cancer" in args.config_file:
    ### train and test
    experiment_set.run_experiment(
        save_model=args.save_model,
        save_result=args.save_result,
        wandb_track=args.wandb_track,
    )


# elif "IDH1_inference" in args.config_file:
elif "IDH1_inference" in args.config_file or "PTEN_inference" in args.config_file:
    ### Infer IDH1-PRKDC
    experiment_set.infer_primpartner()


elif "IDH1_permute" in args.config_file:
    ## IDH1 random permute
    # experiment_set.permut_primpartner(n_sample=1000, 
    #                                 model_path="./experiment/mix_add_GBM/transformer/2024-11-28-11:30:19", 
    #                                 model_type='transformer', cancer='Glioma')
    cancer = config.task.cancer
    for name, model_path in config.model.items():
        experiment_set.permut_primpartner(n_sample=1000, n_iter=20,
                                    model_path=model_path, savename=name, 
                                    model_type='transformer', cancer=cancer)


elif "independent_test" in args.config_file:
    ### independent test
    # experiment_set.independent_test()
    # experiment_set.independent_test(stat=True)
    # experiment_set.independent_test_on_mix()
    # experiment_set.independent_test_on_mix(save_raw=True)
    # experiment_set.independent_test_on_mix(save_raw=True, random_simu=True)
    experiment_set.independent_test_on_mix(save_raw=True, raw_score=True, random_simu=False)

elif "all_inference" in args.config_file:
    experiment_set.infer_all(output_att=False, output_emb=True)

elif "att" in args.config_file:
    experiment_set.get_att()

elif "fewshot" in args.config_file:
    # experiment_set.fewshot_train()
    # experiment_set.fewshot_test(test='mix_fewshot')
    # experiment_set.fewshot_test(test='cancer_specific')

    experiment_set.finetune_GBM()

