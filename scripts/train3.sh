#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw2
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw2) training starts"

# Run the experiments
python train.py --num-epochs 100 --batch-size 32 --optimizer Adam --lr 0.03 --optimizer-params betas=0.9,0.99 --name b_032_lr_0.03
git add -A
git commit -m "auto..."
git push
python train.py --num-epochs 100 --batch-size 64 --optimizer Adam --lr 0.03 --optimizer-params betas=0.9,0.99 --name b_064_lr_0.03
git add -A
git commit -m "auto..."
git push
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.03 --optimizer-params betas=0.9,0.99 --name b_128_lr_0.03
git add -A
git commit -m "auto..."
git push
python train.py --num-epochs 100 --batch-size 256 --optimizer Adam --lr 0.03 --optimizer-params betas=0.9,0.99 --name b_256_lr_0.03
git add -A
git commit -m "auto..."
git push