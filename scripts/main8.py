import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_truncate_and_compare(file1, file2, weights_file=None):
    """
    Load and compare arrays from .npy files, truncating to smallest dimensions
    
    Parameters:
    file1 (str): Path to first .npy file
    file2 (str): Path to second .npy file
    weights_file (str): Optional path to weights .npy file
    
    Returns:
    dict: Results containing arrays, weights, and differences
    """
    # Load arrays
    arr1 = np.load(file1)
    arr2 = np.load(file2)
    
    # Check dimensionality match
    if arr1.ndim != arr2.ndim:
        raise ValueError("Arrays must have the same number of dimensions")
    
    # Calculate minimum dimensions for truncation
    min_shape = tuple(min(s1, s2) for s1, s2 in zip(arr1.shape, arr2.shape))
    
    # Create slice objects for truncation
    slices = tuple(slice(0, m) for m in min_shape)
    
    # Truncate arrays
    arr1_trunc = arr1[slices]
    arr2_trunc = arr2[slices]
    
    # Load and truncate weights if provided
    if weights_file:
        weights = np.load(weights_file)
        # Validate weights dimensions
        if weights.ndim != len(min_shape):
            raise ValueError("Weights must match array dimensionality")
        for i, m in enumerate(min_shape):
            if weights.shape[i] < m:
                raise ValueError(f"Weights dimension {i} too small (needs at least {m})")
        weights_trunc = weights[slices]
    else:
        weights_trunc = np.ones_like(arr1_trunc)
    
    # Calculate differences
    abs_diff = np.abs(arr1_trunc - arr2_trunc)
    weighted_diff = weights_trunc * abs_diff
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Original arrays visualization
    for i, (arr, title) in enumerate(zip([arr1_trunc, arr2_trunc], ["Array 1", "Array 2"])):
        plt.subplot(1, 3, i+1)
        if arr.ndim == 1:
            plt.bar(range(len(arr)), arr)
            plt.title(f"{title}\nShape: {arr.shape}")
        else:
            sns.heatmap(arr, annot=True, cmap='coolwarm', cbar=False)
            plt.title(f"{title}\nShape: {arr.shape}")
    
    # Differences visualization
    plt.subplot(1, 3, 3)
    if weighted_diff.ndim == 1:
        plt.bar(range(len(weighted_diff)), weighted_diff, color='purple')
        plt.title("Weighted Differences\n(1D)")
    else:
        sns.heatmap(weighted_diff, annot=True, cmap='viridis', cbar=True)
        plt.title("Weighted Differences\n(2D)")
    plt.tight_layout()
    
    # Line plot comparison for 1D arrays
    if arr1_trunc.ndim == 1:
        plt.figure(figsize=(10, 5))
        plt.plot(arr1_trunc, label='Array 1', marker='o')
        plt.plot(arr2_trunc, label='Array 2', marker='x')
        plt.plot(weighted_diff, label='Weighted Difference', linestyle='--', marker='s')
        plt.title("Element-wise Comparison")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Element Index")
        plt.ylabel("Value")
        plt.tight_layout()
    
    plt.show()
    
    return {
        'array1': arr1_trunc,
        'array2': arr2_trunc,
        'weights': weights_trunc,
        'absolute_differences': abs_diff,
        'weighted_differences': weighted_diff
    }

# Example usage
if __name__ == "__main__":
    # Update these paths to your .npy files
    results = load_truncate_and_compare(
        file1='P1_C1_data_iqr_zeroed.npy',
        file2='P2_C1_data_iqr_zeroed.npy',
        weights_file=None
    )