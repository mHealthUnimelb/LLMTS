#!/bin/bash
#SBATCH --job-name="time_llm_MIMIC_2"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=80G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate time_llm
#export NCCL_DEBUG=INFO
bash ./scripts/TimeLLM_mTAN_encoder_MIMIC_2gpu_2.sh

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

