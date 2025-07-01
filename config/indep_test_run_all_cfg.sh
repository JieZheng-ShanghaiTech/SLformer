cfg_folder='./config/indep_test_cfg'

for cfg in "$cfg_folder"/*
do
    echo "using config $cfg"
    python slformer/main.py --config_file=$cfg --wandb_track=0 --device=0
done