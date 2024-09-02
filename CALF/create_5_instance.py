import pickle
import numpy as np

with open('datasets/ECG/x_test_dropped_0.7.pkl', 'rb') as file:
    x_test = pickle.load(file)

five_instances = x_test[:5]

print(five_instances.shape)

with open('datasets/ECG/five_dropped_instances.pkl', 'wb') as file:
    pickle.dump(five_instances, file)

print("First five instances have been saved to 'five_dropped_instances.pkl'.")
