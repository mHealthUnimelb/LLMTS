#!/bin/bash
#SBATCH --job-name="plot_heatmap"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate S2IP_LLM
cd Irregular_Classification
python plot_heatmap.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s