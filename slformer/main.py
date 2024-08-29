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
parser.add_argument('--config_file', type=str, default="./config/cancer_specific.yaml",
                    help='config file path')
parser.add_argument('--wandb_track', type=int, default=0,
                    help='whether to track model performance on wandb')
parser.add_argument('--save_model', type=int, default=0,
                    help='whether to save the model checkpoints')
parser.add_argument('--save_result', type=int, default=1,
                    help='whether to save the performance of the models')

parser.add_argument('--n', type=int, default=20,
                    help='genesentence length n')
parser.add_argument('--augmentation', type=str, default=None,
                    help='whether to augment the gene sentence input')
parser.add_argument('--anchor', type=int, default=1,
                    help='whether to include the head anchor gene in a gene sentence')

parser.add_argument('--device', type=int, default=2,
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
parser.add_argument('--mlp_hidden_dim', type=int, default=128,
                    help='hidden dim for MLP')
parser.add_argument('--mlp_output_dim', type=int, default=1,
                    help='output dim for MLP')
# Transformer
parser.add_argument('--d_model', type=int, default=256,
                    help='')
parser.add_argument('--n_head', type=int, default=1,
                    help='')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='')
parser.add_argument('--transformer_hidden_dim', type=int, default=256,
                    help='')
parser.add_argument('--num_layers', type=int, default=2,
                    help='')
parser.add_argument('--add_att', type=int, default=1,
                    help='')
parser.add_argument('--att_nhead', type=int, default=2,
                    help='')
parser.add_argument('--random_init', type=int, default=0,
                    help='')



args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

set_seed(1)

with open(args.data_config_file, 'r') as f:
    data_config = easydict.EasyDict(yaml.safe_load(f))
with open(args.config_file, 'r') as f:
    config = easydict.EasyDict(yaml.safe_load(f))

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


elif "IDH1_inference" in args.config_file:
    ### Infer IDH1-PRKDC
    experiment_set.infer_primpartner()


elif "independent_test" in args.config_file:
    ### independent test
    experiment_set.independent_test()
