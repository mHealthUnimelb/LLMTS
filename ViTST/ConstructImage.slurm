#!/bin/bash
#SBATCH --job-name="ConstructImage"
#SBATCH --account="punim2341"
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --time=18:00:00
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=60G

module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate ViTST

cd dataset/P12data/process_scripts
python ConstructImage.py