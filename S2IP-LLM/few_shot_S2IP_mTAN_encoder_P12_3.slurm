#!/bin/bash
#SBATCH --job-name="S2IP_mTAN_P12_3_few_shot"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=40G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate S2IP_LLM
cd Irregular_Classification
bash scripts/few_shot_S2IP_mTAN_encoder_P12_3.sh

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

