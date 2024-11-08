#!/bin/bash
#SBATCH --job-name="test dropped data"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf

#python main.py --experiment_description Etstcc --run_description run_1 --seed 0 --selected_dataset ECG4 --method TS-TCC
python main.py --experiment_description Etstcc --run_description run_1 --seed 0 --selected_dataset ECG4 --method SimCLR

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

