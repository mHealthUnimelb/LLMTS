#!/bin/bash
#SBATCH --job-name="time_llm"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=80G

module purge
module load foss/2022a
module load mpi4py/3.1.4-Python-3.11.3
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate time_llm
bash ./scripts/Time_ECG_short.sh

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

