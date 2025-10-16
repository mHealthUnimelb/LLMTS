#!/bin/bash
#SBATCH --job-name="few_shot_S2IP_MIMIC_1"
#SBATCH --account="punim2341"
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=80G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate S2IP_LLM
cd Irregular_Classification
bash scripts/few_shot_S2IP_mTAN_encoder_MIMIC_1.sh

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

