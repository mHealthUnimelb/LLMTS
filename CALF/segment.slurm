#!/bin/bash
#SBATCH --job-name="segment"
#SBATCH --account="punim2341"
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=30G

module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf

python segment.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

