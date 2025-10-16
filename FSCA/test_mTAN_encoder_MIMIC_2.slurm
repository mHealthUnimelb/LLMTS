#!/bin/bash
#SBATCH --job-name="FSCA_mTAN_MIMIC_2"
#SBATCH --account="punim2341"
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=80G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate FSCA
cd Classification

CUDA_VISIBLE_DEVICES=0 python src/main_gnn_fsca_mTAN.py \
    --load_model /data/gpfs/projects/punim2341/feixiangz/FSCA/Classification/experiments/_2025-07-24_15-58-23_17A/checkpoints/model_best.pth \
    --wd_project fsca-cla-log00 \
    --batch_size=16 \
    --task_name="classification_mTAN_encoder" \
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
    --model="FSCA_mTAN_encoder_MIMIC_2" \
    --data=MIMIC \
    --seq_len=2880 \
    --feature_dim=96 \
    --optimizer=RAdam \
    --output_dir=experiments/ \
    --patch_size_stride="16 8" \
    --pattern=TRAIN \
    --pos_encoding=learnable \
    --records_file=Classification_records.xls \
    --revin_flag=0 \
    --seed=123 \
    --task=classification \
    --test_flag=0 \
    --val_interval=1 \
    --val_pattern=TEST \
    --w_l2s_flag=0 \
    --w_l2s_v=0.0001 \
    --wandb_flag=0 \
    --quantization 0.016 \
    --classif \
    --num_encoder_heads=1 \
    --learn_emb \

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s