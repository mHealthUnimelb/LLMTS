model_name=UniTS
exp_name=UniTS_MIMIC_1
wandb_mode=disabled
project_name=supervised_learning

random_port=$((RANDOM % 9000 + 1000))

# Supervised learning
torchrun --nnodes 1 --nproc-per-node=1  --master_port $random_port  run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model 128 \
  --des 'Exp' \
  --learning_rate 1e-4 \
  --weight_decay 5e-6 \
  --train_epochs 50 \
  --batch_size 7 \
  --acc_it 32 \
  --debug $wandb_mode \
  --project_name $project_name \
  --clip_grad 100 \
  --pretrained_weight units_x128_pretrain_checkpoint.pth \
  --task_data_config_path data_provider/MIMIC_1.yaml
