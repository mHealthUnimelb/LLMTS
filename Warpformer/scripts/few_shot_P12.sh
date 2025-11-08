python Main_warp.py \
    --data_split_path ../datasets/P12data/splits/phy12_split1.npy \
    --data_path data \
    --quantization 0.016 \
    --percent 10 \
    --batch 32 --lr 1e-3 --epoch 50 --patience 5 \
    --log log/ --save_path save/ \
    --task 'P12' --seed 42 --warp_num '0_0.2_1' \
    --batch_size 32 --d_inner_hid 64 --d_k 8 --d_model 64 --d_v 8 \
    --dropout 0.0 --n_head 1 --n_layers 3

# python Main_warp.py \
#     --data_split_path ../datasets/P12data/splits/phy12_split2.npy \
#     --data_path data \
#     --quantization 0.016 \
#     --percent 10 \
#     --batch 32 --lr 1e-3 --epoch 50 --patience 5 \
#     --log log/ --save_path save/ \
#     --task 'P12' --seed 42 --warp_num '0_0.2_1' \
#     --batch_size 32 --d_inner_hid 64 --d_k 8 --d_model 64 --d_v 8 \
#     --dropout 0.0 --n_head 1 --n_layers 3

# python Main_warp.py \
#     --data_split_path ../datasets/P12data/splits/phy12_split3.npy \
#     --data_path data \
#     --quantization 0.016 \
#     --percent 10 \
#     --batch 32 --lr 1e-3 --epoch 50 --patience 5 \
#     --log log/ --save_path save/ \
#     --task 'P12' --seed 42 --warp_num '0_0.2_1' \
#     --batch_size 32 --d_inner_hid 64 --d_k 8 --d_model 64 --d_v 8 \
#     --dropout 0.0 --n_head 1 --n_layers 3