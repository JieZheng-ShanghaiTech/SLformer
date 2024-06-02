import argparse
import yaml
import easydict

from preprocess import Data_Preprocess
from dataloader import prepare_SL_general_data
from task import Validation_Experiment


parser = argparse.ArgumentParser(description='prepare SL data')

parser.add_argument('--config_file', type=str, default="./config/cross_cancer.yaml",
                    help='config file path')
parser.add_argument('--n', type=int, default=10,
                    help='genesentence length n')
args = parser.parse_args()



with open(args.config_file, 'r') as f:
    config = easydict.EasyDict(yaml.safe_load(f))

data_preprocess = Data_Preprocess(config)
common_data = data_preprocess.get_common_data(n=args.n)


## SL data

experiment_set = Validation_Experiment(
    config=config,
    args = args,
    common_data=common_data,
)

## save SL train/test data

for cancer in config.task.cancer:
    
    data_total = prepare_SL_general_data(
        config=config,
        cancer=cancer,
        common_data=common_data,
    )

    experiment_set.save_train_test_data(data_total, cancer)
