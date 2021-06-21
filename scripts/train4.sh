#!/bin/bash

# Setup env
cd ..
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw2
echo "hello from $(python --version) in $(which python)"
echo "(nlp_hw2) training starts"

# Run the experiments
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.001  --model DependencyParserV2 --model-params transformer_nhead=8 --name 22-6_b_032_lr_0.0010_n_8_glove_v2
python train.py --num-epochs 100 --batch-size 064 --optimizer Adam --lr 0.001  --model DependencyParserV2 --model-params transformer_nhead=8 --name 22-6_b_064_lr_0.0010_n_8_glove_v2
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.001  --model DependencyParserV2 --model-params transformer_nhead=8 --name 22-6_b_128_lr_0.0010_n_8_glove_v2
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0003 --model DependencyParserV2 --model-params transformer_nhead=8 --name 22-6_b_032_lr_0.0003_n_8_glove_v2
python train.py --num-epochs 100 --batch-size 064 --optimizer Adam --lr 0.0003 --model DependencyParserV2 --model-params transformer_nhead=8 --name 22-6_b_064_lr_0.0003_n_8_glove_v2
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.0003 --model DependencyParserV2 --model-params transformer_nhead=8 --name 22-6_b_128_lr_0.0003_n_8_glove_v2
python train.py --num-epochs 100 --batch-size 032 --optimizer Adam --lr 0.0001 --model DependencyParserV2 --model-params transformer_nhead=8 --name 22-6_b_032_lr_0.0001_n_8_glove_v2
python train.py --num-epochs 100 --batch-size 064 --optimizer Adam --lr 0.0001 --model DependencyParserV2 --model-params transformer_nhead=8 --name 22-6_b_064_lr_0.0001_n_8_glove_v2
python train.py --num-epochs 100 --batch-size 128 --optimizer Adam --lr 0.0001 --model DependencyParserV2 --model-params transformer_nhead=8 --name 22-6_b_128_lr_0.0001_n_8_glove_v2
git add -A
git commit -m "auto..."
git push