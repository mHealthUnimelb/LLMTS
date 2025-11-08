input_file="../datasets/ECG/x_test.pkl"
drop_percentage=0.9
action="zero"
output_file="datasets/ECG/x_test_dropped_0.9.pkl"

python random_drop.py $input_file $drop_percentage $action $output_file
