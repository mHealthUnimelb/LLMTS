#!/bin/bash
#SBATCH --job-name="random_drop"
#SBATCH --account="punim2341"
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4

# check that the script is lanuched with sbatch
if ["x$SLURM_JOB_ID" == "x" ]; then
    echo "You need to submit your job to the queuing system with sbatch"
    exit 1
fi

module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf


input_file="datasets/ECG/x_test.pkl"
drop_percentage=0.9
action="zero"
output_file="datasets/ECG/x_test_dropped_0.9.pkl"
python random_drop.py $input_file $drop_percentage $action $output_file

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

