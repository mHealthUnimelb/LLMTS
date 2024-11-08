#!/bin/bash
#SBATCH --job-name="calf"
#SBATCH --account="punim2341"
#BATCH --p gpu-a100-short
#BATCH --time=00:30:00
#BATCH --gres=gpu:1

# check that the script is lanuched with sbatch
if ["x$SLURM_JOB_ID" == "x" ]; then
    echo "You need to submit your job to the queuing system with sbatch"
    exit 1
fi

module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf
python pca.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

