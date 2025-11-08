python Main_warp.py \
    --data_path ../datasets \
    --quantization 0.016 \
    --percent 10 \
    --batch 28 --lr 1e-3 --epoch 50 --patience 5 \
    --log log/ --save_path save/ \
    --task 'MIMIC' --seed 12 --warp_num '0_0.2_1' \
    --batch_size 28 --d_inner_hid 64 --d_k 8 --d_model 64 --d_v 8 \
    --dropout 0.0 --n_head 1 --n_layers 3

# python Main_warp.py \
#     --data_path ../datasets \
#     --quantization 0.016 \
#     --percent 10 \
#     --batch 28 --lr 1e-3 --epoch 50 --patience 5 \
#     --log log/ --save_path save/ \
#     --task 'MIMIC' --seed 123 --warp_num '0_0.2_1' \
#     --batch_size 28 --d_inner_hid 64 --d_k 8 --d_model 64 --d_v 8 \
#     --dropout 0.0 --n_head 1 --n_layers 3

# python Main_warp.py \
#     --data_path ../datasets \
#     --quantization 0.016 \
#     --percent 10 \
#     --batch 28 --lr 1e-3 --epoch 50 --patience 5 \
#     --log log/ --save_path save/ \
#     --task 'MIMIC' --seed 42 --warp_num '0_0.2_1' \
#     --batch_size 28 --d_inner_hid 64 --d_k 8 --d_model 64 --d_v 8 \
#     --dropout 0.0 --n_head 1 --n_layers 3