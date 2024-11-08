import pickle
import numpy as np

with open('datasets/ECG/x_test_dropped_0.8.pkl', 'rb') as file:
    x_test = pickle.load(file)

ten_dropped_instances = x_test[:10]

print(ten_dropped_instances.shape)

with open('datasets/ECG/ten_dropped_instances.pkl', 'wb') as file:
    pickle.dump(ten_dropped_instances, file)

print("First five instances have been saved to 'ten_dropped_instances.pkl'.")
