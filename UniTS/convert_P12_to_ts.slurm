#!/bin/bash
#SBATCH --job-name="units"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --mem=60G

module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf

python get_data.py \
    --classif \
    --data_split_path ./data/P12data/splits/phy12_split3.npy \
    --batch_size 128 \
    --quantization 0.016 \
    --out_dir ./data \
    --base_name P12

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

