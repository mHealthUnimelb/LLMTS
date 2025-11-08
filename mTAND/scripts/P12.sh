module purge
module load foss/2022a
module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate calf

python mTAND_baseline.py \
  --dataset P12
