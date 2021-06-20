#!/bin/bash

# Setup env
cd dependency-parser
conda activate nlp_hw2
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw2) training starts"

# Run the experiments
python train.py --num-epochs 50 --batch-size 032 --optimizer Adam --lr 0.001 --model-params lstm_num_layers=2 lstm_hidden_dim=128 lstm_dropout=0.05 activation_type=lrelu --name 20-6_b_032_lr_0.001_drop_0.05_atype_lrelu
python train.py --num-epochs 50 --batch-size 032 --optimizer Adam --lr 0.001 --model-params lstm_num_layers=2 lstm_hidden_dim=128 lstm_dropout=0.05 activation_type=tanh --name 20-6_b_032_lr_0.001_drop_0.05_atype_relu
git add -A
git commit -m "auto..."
git push