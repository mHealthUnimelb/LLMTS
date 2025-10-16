#!/bin/bash
#SBATCH --job-name="calf"
#SBATCH --account="punim2341"
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=40G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf
export CUDA_LAUNCH_BLOCKING=1

seq_len=2881
model=CALF_mTAN_encoder
pred_len=0

python run_mTAN_encoder.py \
    --root_path ./datasets/P12/ \
    --is_training 1 \
    --task_name classification_mTAN_encoder \
    --model_id P12_$model'_'$seq_len'_'$pred_len \
    --data P12 \
    --data_type irregular \
    --data_split_path ./datasets/P12data/splits/phy12_split3.npy \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 64 \
    --learning_rate 0.00005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 41 \
    --c_out 41 \
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
    --num_ref_points 256 \
    --num_encoder_heads 1 \
    --num_ca_heads 1 \
    --learn_emb \
    --patch_len 16 \
    --stride 8 \

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

