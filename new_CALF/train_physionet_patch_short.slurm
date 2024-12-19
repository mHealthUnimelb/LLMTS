#!/bin/bash
#SBATCH --job-name="calf"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=60G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf

seq_len=2500
model=CALF_2
pred_len=0

python run.py \
    --root_path ./datasets/PhysioNet/ \
    --data_type irregular \
    --is_training 1 \
    --task_name classification \
    --model_id PhysioNet_$model'_'$seq_len'_'$pred_len \
    --data PhysioNet \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 128 \
    --learning_rate 0.00005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 256 \
    --c_out 256 \
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
    --num_ref_points 256 \
    --n 8000 \
    --latent_dim 384 \
    --rec_hidden 128 \
    --quantization 0.016 \
    --classif \
    --num_ca_heads 1 \
    --learn_emb \
    --patch_len 16 \
    --stride 8

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

