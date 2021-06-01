#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw2
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw2) training starts"

# Run the experiments
python train.py --optimizer Adam --lr 1e-3 --optimizer-params betas=0.9,0.99
#git add -A
#git commit -m "auto..."
#git push