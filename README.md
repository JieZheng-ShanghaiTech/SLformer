# SLformer
Gene Sentence Modeling with scRNA-seq Data for Synthetic Lethality Prediction

## preprocess
python slformer/preprocess_sc.py

## prepare SL data
python slformer/prepare_data.py

## pretrain
python slformer/main.py --config_file=./config/pretrain.yaml --wandb_track=0 --save_model=1 --save_result=1 --batch_size=4096 --transformer_hidden_dim=512 --dropout=0.1 --eps=1e-05 --weight_decay=1e-05 --transformer_lr=1e-4 --predictor_lr=1e-4 --mlp_hidden_dim=512 --num_layers=8 --n=50 --random_init=0 --epochs=200 --early_stop=10 --lr_factor=0.5 --lr_patience=3 --device=2

## SL prediction
python slformer/main.py --config_file=./config/cancer_specific.yaml --wandb_track=0 --save_model=0 --save_result=1 --add_att=0 --att_nhead=2 --batch_size=512 --transformer_hidden_dim=256 --dropout=0.1 --eps=1e-05 --weight_decay=1e-05 --transformer_lr=1e-5 --predictor_lr=1e-4 --mlp_hidden_dim=128 --n_head=1 --num_layers=1 --n=10 --device=1

python slformer/main.py --config_file=./config/cross_cancer.yaml --wandb_track=0 --save_model=0 --save_result=1 --add_att=0 --att_nhead=2 --batch_size=512 --transformer_hidden_dim=256 --dropout=0.1 --eps=1e-05 --weight_decay=1e-05 --transformer_lr=1e-5 --predictor_lr=1e-4 --mlp_hidden_dim=128 --n_head=1 --num_layers=1 --n=10 --device=0

python slformer/main.py --config_file=./config/mix.yaml --wandb_track=0 --save_model=0 --save_result=1 --add_att=1 --att_nhead=2 --batch_size=512 --transformer_hidden_dim=256 --dropout=0.1 --eps=1e-05 --weight_decay=1e-05 --transformer_lr=1e-5 --predictor_lr=1e-4 --mlp_hidden_dim=128 --n_head=1 --num_layers=2 --n=10 --device=0