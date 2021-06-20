#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw2
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw2) training starts"

# Run the experiments
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 100 --name 20-6_b_032_lr_0.0001_e_100_base
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 200 --name 20-6_b_032_lr_0.0001_e_200_base
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --random-word-embedding 300 --name 20-6_b_032_lr_0.0001_e_300_base
python train.py --num-epochs 100 --batch-size 064 --optimizer Adam --lr 0.0001 --random-word-embedding 100 --name 20-6_b_064_lr_0.0001_e_100_base
python train.py --num-epochs 100 --batch-size 064 --optimizer Adam --lr 0.0001 --random-word-embedding 200 --name 20-6_b_064_lr_0.0001_e_200_base
python train.py --num-epochs 100 --batch-size 064 --optimizer Adam --lr 0.0001 --random-word-embedding 300 --name 20-6_b_064_lr_0.0001_e_300_base
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.0001 --random-word-embedding 100 --name 20-6_b_128_lr_0.0001_e_100_base
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.0001 --random-word-embedding 200 --name 20-6_b_128_lr_0.0001_e_200_base
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.0001 --random-word-embedding 300 --name 20-6_b_128_lr_0.0001_e_300_base
git add -A
git commit -m "auto..."
git push