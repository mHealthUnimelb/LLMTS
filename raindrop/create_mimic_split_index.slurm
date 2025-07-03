#!/bin/bash
#SBATCH --job-name="raindrop mimic"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=60G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate raindrop

python ./data/MIMICdata/splits/get_split_index.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

