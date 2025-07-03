#!/bin/bash
#SBATCH --job-name="units"
#SBATCH --account="punim2341"
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=100G

module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf

python get_data.py \
    --seed 123 \
    --dataset MIMIC \
    --classif \
    --data_split_path ./MIMIC_split3.npy \
    --batch_size 128 \
    --out_dir ./data \
    --base_name MIMIC

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

