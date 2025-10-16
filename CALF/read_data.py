import pandas as pd

train_data = pd.read_pickle("./datasets/ECG/x_train.pkl")
val_data = pd.read_pickle("./datasets/ECG/x_val.pkl")
test_data = pd.read_pickle("./datasets/ECG/x_test.pkl")
print(len(train_data))
print(len(val_data))
print(len(test_data))

print(len(train_data) + len(val_data) + len(test_data))