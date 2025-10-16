#!/bin/bash
#SBATCH --job-name="S2IP_MIMIC_1"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=100G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate S2IP_LLM
cd Irregular_Classification
bash scripts/few_shot_MIMIC_1.sh

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

