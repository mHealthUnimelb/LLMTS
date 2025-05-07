#!/bin/bash
#SBATCH --job-name="warpformer"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=60G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate warpformer

python Main_warp.py \
    --data_path data \
    --quantization 0.016 \
    --batch 28 --lr 1e-3 --epoch 50 --patience 5 \
    --log log/ --save_path save/ \
    --task 'MIMIC' --seed 12 --warp_num '0_0.2_1' \
    --batch_size 28 --d_inner_hid 64 --d_k 8 --d_model 64 --d_v 8 \
    --dropout 0.0 --n_head 1 --n_layers 3

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
