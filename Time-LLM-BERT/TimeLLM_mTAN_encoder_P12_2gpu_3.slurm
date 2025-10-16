#!/bin/bash
#SBATCH --job-name="time_llm"
#SBATCH --account="punim2341"
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=40G

module purge
module load foss/2022a
module load mpi4py/3.1.4-Python-3.11.3
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate time_llm
#export NCCL_DEBUG=INFO
bash ./scripts/TimeLLM_mTAN_encoder_P12_2gpu_3.sh

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

