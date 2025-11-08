model_name=TimeLLM
train_epochs=100
learning_rate=0.001
llama_layers=6

master_port=2047
num_process=2
batch_size=8
d_model=32
d_ff=128

comment='TimeLLM-PhysioNet'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_ir_classification.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../datasets/P12/ \
  --model_id P12_2 \
  --model $model_name \
  --data P12 \
  --data_split_path ../datasets/P12data/splits/phy12_split2.npy \
  --prompt_domain 1 \
  --features M \
  --seq_len 2881 \
  --num_classes 2 \
  --label_len 0 \
  --pred_len 0 \
  --factor 3 \
  --enc_in 41 \
  --dec_in 41 \
  --c_out 41 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --patch_len 48 \
  --stride 24 \
  --n 8000 \
  --quantization 0.016 \
  --classif \

