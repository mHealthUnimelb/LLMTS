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

python tanenc_classification.py \
    --niters 200 \
    --lr 0.0001 \
    --batch-size 128 \
    --rec-hidden 128 \
    --enc mtan_enc \
    --n 8000 \
    --quantization 0.016 \
    --save 1 \
    --classif \
    --num-heads 1 \
    --learn-emb \
    --dataset physionet \
    --seed 2021