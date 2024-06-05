import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from hp5dataset import MultiFileDataset as Dataset
import json

def calculate_elementwise_variance(data):
    """
    Calculate the variance element-wise for a list of lists.

    Parameters:
    data (list of lists): The input data where each sublist represents a dataset row.

    Returns:
    list: A list containing the variance of each element across the sublists.
    """
    # Convert the list of lists to a numpy array
    np_data = np.array(data)
    
    # Calculate variance along the columns (axis=0)
    variances = np.var(np_data)
    
    return variances.tolist()

def calculate_centers(data):
    # Calculate the center of the bounding boxes
    # Bounding box format is assumed to be [x_min, y_min, x_max, y_max]
    bbox = data  # Assuming bbox is directly accessible like this
    center_x = (bbox[:, 0] + bbox[:, 2]) / 2
    center_y = (bbox[:, 1] + bbox[:, 3]) / 2
    return center_x, center_y

def plot_grid(percentages):
    """
    Plot a heatmap of percentages on a grid, displaying only the cell colors and percentage values.

    Parameters:
    percentages (list of lists): The input data where each sublist represents a row in the grid.
                                 Each element should be a percentage.
    """
    # Ensure percentages is a numpy array for easier manipulation
    percentages = np.array(percentages)
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    # Create a heatmap
    cax = ax.matshow(percentages, cmap='viridis')  # Using 'viridis' for good color contrast
    
    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Display percentage values in each cell
    for (i, j), val in np.ndenumerate(percentages):
        # Get the color of the cell
        cell_color = cax.cmap(cax.norm(val))
        # Calculate the brightness of the color
        brightness = cell_color[0]*0.299 + cell_color[1]*0.587 + cell_color[2]*0.114  # Luminance formula
        text_color = 'white' if brightness < 0.5 else 'black'
        
        ax.text(j, i, f'{val:.2f}%', ha='center', va='center', color=text_color, fontweight='bold', fontsize=16)
    
    plt.show()
    plt.savefig('grid2.png')

def section_distribution(loader):
    counts = np.zeros((5, 3))
    total_points = 0

    for data in tqdm(loader, total=len(loader), desc="Processing"):
        center_x, center_y = calculate_centers(data[4])
        for x, y in zip(center_x, center_y):
            try:
                column = int(x * 3)
                row = int(y * 5)
                if column == 3:  # Handle edge case where center_x == 1
                    column = 2
                if row == 5:  # Handle edge case where center_y == 1
                    row = 4
                counts[row, column] += 1
                total_points += 1
            except:
                print(f"Error at {row}, {column}")

    percentages = (counts / total_points) * 100
    return percentages

def get_distribution():
    data_dir = '/data2/peter/auto_dataset/dataset/output'
    dataset = Dataset(data_dir=data_dir, type="test", csv_file=None, decode_type="point")
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    percentages = section_distribution(loader)
    # percentages = [[21.28, 4.02, 9.57] ,[6.79, 5.84, 3.16], [4.72, 6.00, 2.31], [3.63, 7.31, 2.50], [11.02, 6.85, 4.98]]
    print(calculate_elementwise_variance(percentages))
    print(percentages)
    plot_grid(percentages)

def count_different_apps():
    json_dir = "/data2/peter/auto_dataset/dataset/action.json"
    with open(json_dir, 'r') as file:
        data = json.load(file)
    
    different_app = set()
    for gif in data.keys():
        different_app.add(data[gif]['app'])
    print(len(different_app))

def count_app_names():
    directory = "/data/peter/animation/dataset"
    # Create a counter to store app names
    different_app = set()
    
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            # Extract the app name which is the substring before "_trace"
            app_name = filename.split('_trace')[0]
            different_app.add(app_name)
    
    print(len(different_app))

def count_dataset_files():
    data_dir = '/data2/peter/validation_set/merge'
    dataset = Dataset(data_dir=data_dir, type="test", csv_file=None, decode_type="point")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(len(loader))

if __name__ == "__main__":
    count_dataset_files()
        
