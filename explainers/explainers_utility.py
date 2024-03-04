import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from constants import CAVE_COLS, CAVE_ROWS, MARIO_COLS, MARIO_ROWS, SUPERCAT_COLS, SUPERCAT_ROWS

def process(value):
    percentile_threshold = np.percentile(value, 80)  # Top 20% threshold
    binary_map = (value > percentile_threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    try:
        np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    except ValueError:
        return None
    largest_area_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    largest_area_indices = np.argwhere(labels == largest_area_label)
    return largest_area_indices

def create_array_with_largest_area(value):
    """
    Create an array with 1s in indices of the largest area and 10s for the rest.
    
    Parameters:
        heatmap (ndarray): The original heatmap array.
        largest_area_indices (ndarray): Indices of the largest area.
    
    Returns:
        ndarray: New array with 1s in indices of the largest area and 10s for the rest.
    """
    # Create a new array filled with 10s
    new_array = np.full_like(value, fill_value=10, dtype=int)
    largest_area_indices = process(value)
    # Set 1s at indices of the largest area
    if largest_area_indices.all() != None:
        for idx in largest_area_indices:
            new_array[idx[0], idx[1]] = 1
    return new_array
