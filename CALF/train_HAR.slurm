#!/bin/bash
#SBATCH --job-name="calf"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=80G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf
export CUDA_LAUNCH_BLOCKING=1

seq_len=221
model=CALF
pred_len=0

python run.py \
    --seed 123 \
    --root_path ./datasets/activity/ \
    --is_training 1 \
    --task_name classification \
    --model_id Activity_$model'_'$seq_len'_'$pred_len \
    --data activity \
    --data_type irregular \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 128 \
    --learning_rate 0.005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 12 \
    --c_out 12 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --patience 10 \
    --task_loss ce \
    --word_embedding_path wte_pca_600.pt \
    --task_w 1.0 \
    --feature_w 0.01 \
    --output_w 1.0 \
    --classif \
    --classify-pertp

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

