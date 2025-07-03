#!/bin/bash
#SBATCH --job-name="mTAND"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=80G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf

python mTAND_baseline.py \
  --lr 0.001 \
  --batch-size 256 \
  --rec-hidden 512 \
  --enc mtan_enc_activity \
  --dataset activity \
  --classify-pertp

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s