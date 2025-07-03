#!/bin/bash
#SBATCH --job-name="FSCA"
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

CUDA_VISIBLE_DEVICES=0 python src/main_gnn_fsca.py --wd_project fsca-cla-log00 --batch_size=64 --comment="classification from Scratch" --d_ff=512 --d_l_comp=512 --d_model=768 --data_class=tsra --dropout=0.1 --epochs=120 --key_metric=accuracy --layer_index=4*0_1_4*0_1_4 --lr=0.001 --model=FSCA --name=JapaneseVowels --optimizer=RAdam --output_dir=experiments/ --patch_size_stride="4 2" --pattern=TRAIN --pos_encoding=learnable --records_file=Classification_records.xls --revin_flag=0 --seed=42 --task=classification --test_flag=0 --val_interval=1 --val_pattern=TEST --w_l2s_flag=0 --w_l2s_v=0.0001 --wandb_flag=0
