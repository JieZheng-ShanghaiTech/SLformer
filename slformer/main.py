import argparse
import torch
import yaml
import easydict

from util import set_seed
from preprocess import Data_Preprocess
from task import Validation_Experiment



parser = argparse.ArgumentParser(description='SL prediction')

parser.add_argument('--config_file', type=str, default="./config/cancer_specific.yaml",
                    help='config file path')
parser.add_argument('--n', type=int, default=10,
                    help='genesentence length n')

parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=120,
                    help='number of epochs to train')
parser.add_argument('--early_stop', type=int, default=5,
                    help="Early stopping patience")
# optimizer
parser.add_argument('--lr', type=float, default=5e-5,
                    help='learning rate (default: 5e-5)')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.99),
                    help='')
parser.add_argument('--eps', type=float, default=1e-05,
                    help='')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='')
# MLP
parser.add_argument('--input_dim', type=int, default=256*2,
                    help='input dim for MLP')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden dim for MLP')
parser.add_argument('--output_dim', type=int, default=1,
                    help='output dim for MLP')
# Transformer
parser.add_argument('--d_model', type=int, default=256,
                    help='')
parser.add_argument('--n_head', type=int, default=1,
                    help='')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='')
parser.add_argument('--dim_feedforward', type=int, default=256*2,
                    help='')
parser.add_argument('--num_layers', type=int, default=1,
                    help='')
parser.add_argument('--add_att', type=bool, default=False,
                    help='')
parser.add_argument('--att_nhead', type=int, default=2,
                    help='')




args = parser.parse_args()

# device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
args.device = torch.device("cuda:" + str(args.device)) 

set_seed(1)


with open(args.config_file, 'r') as f:
    config = easydict.EasyDict(yaml.safe_load(f))

data_preprocess = Data_Preprocess(config)
common_data = data_preprocess.get_common_data(n=args.n)



experiment_set = Validation_Experiment(
    config=config,
    args = args,
    common_data=common_data,
)

experiment_set.run_experiment(
    save_model=False,
    save_log=True,
)


# experiment_set.get_benchmark_data(
#     data_total=data_total,
#     cancer_type=CANCER
# )

# experiment_set.infer_primpartner(
#     data_fname="IDH1_reactome_Glioma",
#     output_dir="/home/jienihu/sc/GeneSentence/result/IDH1/transformer_train"
# )

# # calculate random AUC/AUPR
# experiment_set.get_random_auc()