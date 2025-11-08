seq_len=2500
model=CALF
pred_len=0

python run.py \
    --seed 42 \
    --root_path ../datasets/ECG/ \
    --is_training 1 \
    --task_name classification \
    --model_id ECG_$model'_'$seq_len'_'$pred_len \
    --data ECG \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 128 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
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
    --output_w 1.0

