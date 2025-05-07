#!/bin/bash
#SBATCH --job-name="mTAN"
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
cd src

python3 tanenc_classification.py \
    --niters 1000 \
    --lr 0.001 \
    --batch-size 256 \
    --rec-hidden 512 \
    --enc mtan_enc_activity  \
    --quantization 0.016 \
    --save 1 \
    --classif \
    --num-heads 1 \
    --learn-emb \
    --dataset activity \
    --seed 0 \
    --classify-pertp