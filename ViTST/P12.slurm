#!/bin/bash
#SBATCH --job-name="ViTSTS"
#SBATCH --account="punim2341"
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=60G

module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate ViTST

cd code/Vision-Text/
sh vtcls_script.sh
