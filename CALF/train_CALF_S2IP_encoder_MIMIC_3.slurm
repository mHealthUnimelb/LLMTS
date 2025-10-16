#!/bin/bash
#SBATCH --job-name="calf_s2ip"
#SBATCH --account="punim2341"
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --nodes=1
#SBATCH --time=80:00:00
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

seq_len=2880
model=CALF_S2IP_encoder
pred_len=0

python run.py \
    --seed 42 \
    --root_path ./datasets/MIMIC/ \
    --is_training 1 \
    --task_name classification \
    --model_id MIMIC_42_$model'_'$seq_len'_'$pred_len \
    --data MIMIC \
    --data_type irregular \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.00005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 96 \
    --c_out 96 \
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
    --quantization 0.016 \
    --classif \
    --patch_len 256 \
    --stride 128 \
    --trend_length 96 \
    --seasonal_length 96 \

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

