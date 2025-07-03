import pickle
import numpy as np

with open('datasets/ECG/x_test_dropped_0.5.pkl', 'rb') as file:
    x_test = pickle.load(file)

# with open('datasets/ECG/x_test.pkl', 'rb') as file:
#     x_test = pickle.load(file)

with open('datasets/ECG/state_test.pkl', 'rb') as file:
    state_test = pickle.load(file)

index = [5, 1000, 888, 623, 40, 100, 150, 200, 300, 366]
ten_dropped_instances = x_test[index]
ten_dropped_labels = state_test[index]

print(ten_dropped_instances.shape)
print(ten_dropped_labels.shape)

with open('datasets/ECG/ten_dropped_0.5_instances.pkl', 'wb') as file:
    pickle.dump(ten_dropped_instances, file)

# with open('datasets/ECG/ten_instances.pkl', 'wb') as file:
#     pickle.dump(ten_dropped_instances, file)

with open('datasets/ECG/ten_dropped_labels.pkl', 'wb') as file:
    pickle.dump(ten_dropped_labels, file)

print("First five instances have been saved to 'ten_dropped_instances.pkl'.")
# print("First five instances have been saved to 'ten_instances.pkl'.")