model_name=TimeLLM_4
train_epochs=2
learning_rate=0.001
llama_layers=6

master_port=2061
num_process=1
batch_size=16
d_model=24
d_ff=128

comment='TimeLLM-forecast-physionet_test'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_ir_forecasting.py \
  --task_name ir_forecast \
  --is_training 1 \
  --root_path ./dataset/physionet/ \
  --model_id physionet_test \
  --model $model_name \
  --data physionet \
  --prompt_domain 1 \
  --features M \
  --seq_len 512 \
  --pred_len 512 \
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
  --patch_len 16 \
  --stride 8 \

#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_512_192 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 512 \
#  --label_len 48 \
#  --pred_len 192 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1 \
#  --d_model 32 \
#  --d_ff 128 \
#  --batch_size $batch_size \
#  --learning_rate 0.02 \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --model_comment $comment
#
#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_512_336 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 512 \
#  --label_len 48 \
#  --pred_len 336 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --batch_size $batch_size \
#  --lradj 'COS'\
#  --learning_rate 0.001 \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --model_comment $comment
#
#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_512_720 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 512 \
#  --label_len 48 \
#  --pred_len 720 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --batch_size $batch_size \
#  --learning_rate $learning_rate \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --model_comment $comment
