#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw2
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw2) training starts"

# Run the experiments
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 100 --model-params lstm_num_layers=3 lstm_hidden_dim=128 --name 20-6_b_032_e_100_l_3_h_128_base
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 200 --model-params lstm_num_layers=4 lstm_hidden_dim=128 --name 20-6_b_032_e_100_l_4_h_128_base
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 300 --model-params lstm_num_layers=5 lstm_hidden_dim=128 --name 20-6_b_032_e_100_l_5_h_128_base
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 100 --model-params lstm_num_layers=3 lstm_hidden_dim=256 --name 20-6_b_032_e_100_l_3_h_256_base
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 200 --model-params lstm_num_layers=4 lstm_hidden_dim=256 --name 20-6_b_032_e_100_l_4_h_256_base
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 300 --model-params lstm_num_layers=5 lstm_hidden_dim=256 --name 20-6_b_032_e_100_l_5_h_256_base
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 100 --model-params lstm_num_layers=3 lstm_hidden_dim=512 --name 20-6_b_032_e_100_l_3_h_512_base
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 200 --model-params lstm_num_layers=4 lstm_hidden_dim=512 --name 20-6_b_032_e_100_l_4_h_512_base
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 300 --model-params lstm_num_layers=5 lstm_hidden_dim=512 --name 20-6_b_032_e_100_l_5_h_512_base
git add -A
git commit -m "auto..."
git push