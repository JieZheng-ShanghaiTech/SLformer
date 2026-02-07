import argparse
import yaml
import easydict
import logging

from util import set_seed
from task import EXPERIMENT_REGISTRY
from task import Training_Models, Inference
from preprocess import Data_Preprocess


parser = argparse.ArgumentParser(description='SL prediction')

parser.add_argument('--data_config_file', type=str, required=True,
                    help='data preprocess config file path')
parser.add_argument('--config_file', type=str, required=True,
                    help='experiment config file path')
parser.add_argument('--wandb_track', type=int, default=0,
                    help='whether to track model performance on wandb')
parser.add_argument('--save_model', type=int, default=1,
                    help='whether to save the model checkpoints')
parser.add_argument('--save_result', type=int, default=1,
                    help='whether to save the performance of the models')

parser.add_argument('--n', type=int, default=10,
                    help='genesentence length')
parser.add_argument('--add_kg', type=int, default=1,
                    help='whether to concatenate with knowledge graph embedding')

parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=200,
                    help='training epochs')
parser.add_argument('--early_stop', type=int, default=5,
                    help="Early stopping patience")
# these are used for SLformer model only
parser.add_argument('--transformer_lr', type=float, default=1e-5,
                    help='learning rate for transformer encoder (default: 5e-5)')
parser.add_argument('--predictor_lr', type=float, default=1e-4,
                    help='learning rate for MLP predictor (default: 1e-4)')
## optimizer params
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
# MLP predictor
parser.add_argument('--mlp_input_dim', type=int, default=256*2,
                    help='input dim for MLP')
parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                    help='hidden dim for MLP')
parser.add_argument('--mlp_output_dim', type=int, default=1,
                    help='output dim for MLP')
# Transformer encoder
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
            
    
experiment = args.config_file.split('/')[-1].split('.yaml')[0]
if experiment not in EXPERIMENT_REGISTRY:
    raise ValueError(f"Invalid experiment setting for {experiment}. Please choose from {', '.join(EXPERIMENT_REGISTRY.keys())}")
experiment_mode = EXPERIMENT_REGISTRY[experiment]


data_preprocess = Data_Preprocess(data_config)
common_data = data_preprocess.get_common_data(sent_n=200)


if experiment_mode == "train":
    experiment_set = Training_Models(
        config=config,
        args = args,
        common_data=common_data,
    )
    experiment_set.run_experiment(
        save_model=args.save_model,
        save_result=args.save_result,
        wandb_track=args.wandb_track,
    )
elif experiment_mode == "inference":
    experiment_set = Inference(
        config=config,
        args = args,
        common_data=common_data,
    )
    experiment_set.run_experiment(experiment)



