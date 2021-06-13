#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw2
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw2) training starts"

# Run the experiments
python train.py --num-epochs 150 --batch-size 32 --optimizer Adam --lr 0.001 --optimizer-params betas=0.9,0.99 --name 13-6_b_032_lr_0.001_adam
python train.py --num-epochs 150 --batch-size 64 --optimizer Adam --lr 0.001 --optimizer-params betas=0.9,0.99 --name 13-6_b_064_lr_0.001_adam
python train.py --num-epochs 150 --batch-size 128 --optimizer Adam --lr 0.001 --optimizer-params betas=0.9,0.99 --name 13-6_b_128_lr_0.001_adam
python train.py --num-epochs 150 --batch-size 256 --optimizer Adam --lr 0.001 --optimizer-params betas=0.9,0.99 --name 13-6_b_256_lr_0.001_adam
git add -A
git commit -m "auto..."
git push