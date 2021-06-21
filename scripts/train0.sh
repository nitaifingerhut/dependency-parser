#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw2
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw2) training starts"

# Run the experiments
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0010 --name 22-6_b_032_lr_0.0010_glove_v1
python train.py --num-epochs 100 --batch-size 064 --optimizer Adam --lr 0.0010 --name 22-6_b_064_lr_0.0010_glove_v1
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.0010 --name 22-6_b_128_lr_0.0010_glove_v1
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0003 --name 22-6_b_032_lr_0.0003_glove_v1
python train.py --num-epochs 100 --batch-size 064 --optimizer Adam --lr 0.0003 --name 22-6_b_064_lr_0.0003_glove_v1
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.0003 --name 22-6_b_128_lr_0.0003_glove_v1
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --name 22-6_b_032_lr_0.0001_glove_v1
python train.py --num-epochs 100 --batch-size 064 --optimizer Adam --lr 0.0001 --name 22-6_b_064_lr_0.0001_glove_v1
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.0001 --name 22-6_b_128_lr_0.0001_glove_v1
git add -A
git commit -m "auto..."
git push