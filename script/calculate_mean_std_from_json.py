import json
import numpy as np
import sys

def calculate_mean_std_from_json(file_path):
    # Load JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract the third element (index 2) from each sublist
    values = [item[2] for item in data]
    
    # Calculate mean and standard deviation
    mean = np.mean(values)
    std = np.std(values)
    
    return mean, std

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_mean_std_from_json.py <path_to_json_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    mean, std = calculate_mean_std_from_json(file_path)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
