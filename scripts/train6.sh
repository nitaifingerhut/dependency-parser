#!/bin/bash

# Setup env
cd ..
conda activate nlp_hw2
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw2) training starts"
cd dependency-parser

# Run the experiments
python train.py --num-epochs 35 --batch-size 32  --optimizer Adam --lr 0.0001 --optimizer-params betas=0.9,0.99 --name 19-6_b_032_lr_0.0001_base_new --random_word_embedding 1
git add -A
git commit -m "auto..."
git push