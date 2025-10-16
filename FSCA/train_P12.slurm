#!/bin/bash
#SBATCH --job-name="FSCA_ECG"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --time=04:00:00
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

CUDA_VISIBLE_DEVICES=0 python src/main_gnn_fsca_ir.py \
    --wd_project fsca-cla-log00 \
    --batch_size=64 \
    --comment="classification from Scratch" \
    --training_flag=1 \
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
    --data=P12 \
    --data_split_path /data/gpfs/projects/punim2341/feixiangz/CALF/datasets/P12data/splits/phy12_split3.npy \
    --seq_len=2881 \
    --feature_dim=41 \
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
    --wandb_flag=0 \
    --quantization 0.016 \
    --classif \

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s