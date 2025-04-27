import numpy as np

# Load the numpy arrays for Task 2 and Task 4
task2_output = np.load('/zhome/a2/c/213547/DLCV/adlcv_project/adlcv_project-master/data/processed_text/stride_224_text/scenarios_npy/mario-1-2.npy')
task4_output = np.load('/zhome/a2/c/213547/DLCV/adlcv_project/adlcv_project-master/src/task_4/generated_sequences/generated_frame_1.npy')

# Print the shape and a few values for Task 2
print("Task 3 Output:")
print(f"Shape: {task2_output.shape}")
print(f"Sample values (first 5 pixels of the first frame): \n{task2_output[0, :, :5]}")  # Adjust indices for slicing

# Print the shape and a few values for Task 4
print("\nTask 4 Output:")
print(f"Shape: {task4_output.shape}")
print(f"Sample values (first 5 pixels of the first frame): \n{task4_output[0, :, :5]}")  # Adjust indices for slicing
