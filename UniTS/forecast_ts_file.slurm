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
conda activate units

python get_forecast_ts_file.py \
    --seed 42 \
    --batch_size 64 \
    --problemname P12_1 \
    --out_dir ./data/forecast/physionet_1 \
    --out_filename physionet_1 \

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

