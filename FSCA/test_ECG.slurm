#!/bin/bash
#SBATCH --job-name="FSCA_ECG"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=40G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate FSCA
cd Classification

CUDA_VISIBLE_DEVICES=0 python src/main_gnn_fsca_ecg.py \
    --load_model /data/gpfs/projects/punim2341/feixiangz/FSCA/Classification/experiments/_2025-07-02_00-06-48_Nd3/checkpoints/model_best.pth \
    --wd_project fsca-cla-log00 \
    --batch_size=64 \
    --comment="classification from Scratch" \
    --training_flag=0 \
    --test_only testset \
    --d_ff=512 \
    --d_l_comp=512 \
    --d_model=768 \
    --data_class=tsra \
    --dropout=0.1 \
    --epochs=50 \
    --key_metric=accuracy \
    --layer_index=6*0_1_4*0_1_4 \
    --lr=0.001 \
    --model=FSCA \
    --root_path /data/gpfs/projects/punim2341/feixiangz/CALF/datasets/ECG/ \
    --data_path x_test_dropped_0.9.pkl \
    --optimizer=RAdam \
    --output_dir=experiments/ \
    --patch_size_stride="16 8" \
    --pattern=TRAIN \
    --pos_encoding=learnable \
    --records_file=Classification_records.xls \
    --revin_flag=0 \
    --seed=2024 \
    --task=classification \
    --test_flag=0 \
    --val_interval=1 \
    --val_pattern=TEST \
    --w_l2s_flag=0 \
    --w_l2s_v=0.0001 \
    --wandb_flag=0

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s