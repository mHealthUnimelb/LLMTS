#!/bin/bash
#SBATCH --job-name="warpformer"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=40G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate warpformer

python Main_warp.py \
    --data_path data \
    --batch 64 --lr 1e-3 --epoch 50 --patience 5 \
    --log log/ \
    --save_path save/ \
    --task 'active' --seed 42 --warp_num '1.4_0.2_1' 

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
