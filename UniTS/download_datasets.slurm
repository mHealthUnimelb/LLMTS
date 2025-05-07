#!/bin/bash
#SBATCH --job-name="units-P12-1"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=40G

bash download_data_all.sh

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s