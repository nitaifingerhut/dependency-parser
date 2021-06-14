#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw2
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw2) training starts"

# Run the experiments
python train.py --num-epochs 100 --batch-size 32  --optimizer Adam --lr 0.0001 --optimizer-params betas=0.9,0.99 --model-params lstm_dropout=0.2 --name 14-6_b_032_lr_0.0001_d_0.2
python train.py --num-epochs 100 --batch-size 64  --optimizer Adam --lr 0.0001 --optimizer-params betas=0.9,0.99 --model-params lstm_dropout=0.2 --name 14-6_b_064_lr_0.0001_d_0.2
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.0001 --optimizer-params betas=0.9,0.99 --model-params lstm_dropout=0.2 --name 14-6_b_128_lr_0.0001_d_0.2
python train.py --num-epochs 100 --batch-size 32  --optimizer Adam --lr 0.0001 --optimizer-params betas=0.9,0.99 --model-params lstm_dropout=0.3 --name 14-6_b_032_lr_0.0001_d_0.3
python train.py --num-epochs 100 --batch-size 64  --optimizer Adam --lr 0.0001 --optimizer-params betas=0.9,0.99 --model-params lstm_dropout=0.3 --name 14-6_b_064_lr_0.0001_d_0.3
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.0001 --optimizer-params betas=0.9,0.99 --model-params lstm_dropout=0.3 --name 14-6_b_128_lr_0.0001_d_0.3
python train.py --num-epochs 100 --batch-size 32  --optimizer Adam --lr 0.0001 --optimizer-params betas=0.9,0.99 --model-params lstm_dropout=0.3 --name 14-6_b_032_lr_0.0001_d_0.4
python train.py --num-epochs 100 --batch-size 64  --optimizer Adam --lr 0.0001 --optimizer-params betas=0.9,0.99 --model-params lstm_dropout=0.3 --name 14-6_b_064_lr_0.0001_d_0.4
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.0001 --optimizer-params betas=0.9,0.99 --model-params lstm_dropout=0.3 --name 14-6_b_128_lr_0.0001_d_0.4
git add -A
git commit -m "auto..."
git push