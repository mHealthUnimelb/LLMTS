#!/bin/bash
#SBATCH --job-name="moment"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=40G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate moment

python run.py \
  --seed 12 \
  --task_name classification \
  --is_training 1 \
  --data MIMIC \
  --root_path ./data/MIMIC \
  --data_path MIMIC \
  --data_split_path ./MIMIC_split2.npy \
  --model_id MIMIC \
  --seq_len 2880 \
  --num_variables 96 \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 32 \
  --patience 10 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --loss 'CE' \
  --lradj type1 \
  --num_classes 2 \
  --classif


##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

