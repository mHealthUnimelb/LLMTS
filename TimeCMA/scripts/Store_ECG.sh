 #!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH

data_name="ECG"
root_path=("/data/gpfs/projects/punim2341/feixiangz/datasets/ECG/1000/")
data_paths=("ECG")
divides=("train" "val" "test")
num_nodes=2
input_len=350
output_len=0

# for data_path in "${data_paths[@]}"; do
#   for divide in "${divides[@]}"; do
#     log_file="./Results/emb_logs/ECG_${divide}.log"
#     nohup python storage/store_emb.py --divide $divide --root_path $root_path --data_path $data_path --training_flag 1 --device $device --num_nodes $num_nodes --seq_len 2500 --input_len $input_len --output_len $output_len > $log_file &
#   done
# done

divide="train"
data_path="ECG"
root_path="/data/gpfs/projects/punim2341/feixiangz/datasets/ECG/1000/"
python storage/store_emb.py --data_name $data_name --divide $divide --root_path $root_path --data_path $data_path --training_flag 1 --num_nodes $num_nodes --seq_len 2500 --input_len $input_len --output_len $output_len