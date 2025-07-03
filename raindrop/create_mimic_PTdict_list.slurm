#!/bin/bash
#SBATCH --job-name="raindrop mimic"
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
conda activate raindrop

python ./data/MIMICdata/process_scripts/mimic_to_PTdict.py \
  --in_file ./data/MIMICdata/process_scripts/mimic.pt \
  --out_dir ./data/MIMICdata/processed_data

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

