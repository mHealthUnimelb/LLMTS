#!/bin/bash
#SBATCH --job-name="extract wte"
#SBATCH --account="punim2341"
#BATCH --p gpu-a100-short
#BATCH --time=00:30:00
#BATCH --gres=gpu:1

module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf
python extract_wte.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

