model_name=TimeLLM
train_epochs=100
learning_rate=0.001
llama_layers=6

master_port=2047
num_process=1
batch_size=32
d_model=32
d_ff=128

comment='TimeLLM-ECG'

# train
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_classification.py \
 --task_name classification \
 --is_training 1 \
 --root_path ../datasets/ECG/ \
 --data_path x_train.pkl \
 --model_id ECG \
 --model $model_name \
 --data ECG \
 --features M \
 --seq_len 2500 \
 --num_classes 4 \
 --label_len 0 \
 --pred_len 0 \
 --factor 3 \
 --enc_in 2 \
 --dec_in 2 \
 --c_out 2 \
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
 --model_comment $comment

# test
# accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_classification.py \
#   --task_name classification \
#   --is_training 0 \
#   --root_path ../datasets/ECG/ \
#   --data_path x_test_dropped_0.9.pkl \
#   --model_id ECG \
#   --model $model_name \
#   --data ECG \
#   --prompt_domain 1 \
#   --features M \
#   --seq_len 2500 \
#   --num_classes 4 \
#   --label_len 0 \
#   --pred_len 0 \
#   --factor 3 \
#   --enc_in 2 \
#   --dec_in 2 \
#   --c_out 2 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --llm_model GPT2 \
#   --llm_dim 768 \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment