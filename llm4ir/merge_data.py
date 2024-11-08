import pickle
import torch


def merge_and_save_datasets(x_test_path, state_test_path, output_path):
    """
    Merges datasets from x_test.pkl and state_test.pkl and saves them as a .pt file.

    Args:
    - x_test_path (str): Path to the x_test.pkl file.
    - state_test_path (str): Path to the state_test.pkl file.
    - output_path (str): Path where the merged data will be saved as a .pt file.
    """
    # Step 1: Load the datasets from the .pkl files
    with open(x_test_path, 'rb') as f:
        x_test = pickle.load(f)

    with open(state_test_path, 'rb') as f:
        state_test = pickle.load(f)

    # Step 2: Merge the datasets into a single dictionary
    merged_data = {
        'samples': x_test,
        'labels': state_test
    }

    # Step 3: Save the merged data as a .pt file
    torch.save(merged_data, output_path)
    print(f"Merged data saved as {output_path}")


# Example usage
merge_and_save_datasets('data/ECG4/x_train.pkl', 'data/ECG4/state_train.pkl', 'data/ECG4/train.pt')
merge_and_save_datasets('data/ECG4/x_val.pkl', 'data/ECG4/state_val.pkl', 'data/ECG4/val.pt')
merge_and_save_datasets('data/ECG4/x_test.pkl', 'data/ECG4/state_test.pkl', 'data/ECG4/test.pt')

merge_and_save_datasets('data/ECG4/x_test_dropped_0.1.pkl', 'data/ECG4/state_test.pkl', 'data/ECG4/test_dropped_0.1.pt')
merge_and_save_datasets('data/ECG4/x_test_dropped_0.2.pkl', 'data/ECG4/state_test.pkl', 'data/ECG4/test_dropped_0.2.pt')
merge_and_save_datasets('data/ECG4/x_test_dropped_0.3.pkl', 'data/ECG4/state_test.pkl', 'data/ECG4/test_dropped_0.3.pt')
merge_and_save_datasets('data/ECG4/x_test_dropped_0.4.pkl', 'data/ECG4/state_test.pkl', 'data/ECG4/test_dropped_0.4.pt')
merge_and_save_datasets('data/ECG4/x_test_dropped_0.5.pkl', 'data/ECG4/state_test.pkl', 'data/ECG4/test_dropped_0.5.pt')
merge_and_save_datasets('data/ECG4/x_test_dropped_0.6.pkl', 'data/ECG4/state_test.pkl', 'data/ECG4/test_dropped_0.6.pt')
merge_and_save_datasets('data/ECG4/x_test_dropped_0.7.pkl', 'data/ECG4/state_test.pkl', 'data/ECG4/test_dropped_0.7.pt')
merge_and_save_datasets('data/ECG4/x_test_dropped_0.8.pkl', 'data/ECG4/state_test.pkl', 'data/ECG4/test_dropped_0.8.pt')
merge_and_save_datasets('data/ECG4/x_test_dropped_0.9.pkl', 'data/ECG4/state_test.pkl', 'data/ECG4/test_dropped_0.9.pt')