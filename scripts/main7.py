import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_compare(file1, file2, weights_file=None):
    """
    Load two numpy arrays from .npy files and perform weighted comparison
    
    Parameters:
    file1 (str): Path to first .npy file
    file2 (str): Path to second .npy file
    weights_file (str): Optional path to weights .npy file
    """
    # Load arrays
    arr1 = np.load(file1)
    arr2 = np.load(file2)
    
    # Validate shapes
    if arr1.shape != arr2.shape:
        raise ValueError(f"Array shapes must match: {arr1.shape} vs {arr2.shape}")
    
    # Load or create weights
    weights = np.load(weights_file) if weights_file else np.ones_like(arr1)
    
    if weights.shape != arr1.shape:
        raise ValueError(f"Weights shape {weights.shape} must match array shapes {arr1.shape}")

    # Calculate differences
    abs_diff = np.abs(arr1 - arr2)
    weighted_diff = weights * abs_diff
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot original arrays
    plt.subplot(1, 3, 1)
    sns.heatmap(arr1, annot=True, cmap='coolwarm', cbar=False)
    plt.title("Array 1")
    
    plt.subplot(1, 3, 2)
    sns.heatmap(arr1, annot=True, cmap='coolwarm', cbar=False)
    plt.title("Array 2")
    
    # Plot weighted differences
    plt.subplot(1, 3, 3)
    sns.heatmap(weighted_diff, annot=True, cmap='viridis', cbar=True)
    plt.title("Weighted Differences")
    
    plt.tight_layout()
    
    # Create line plot comparison for 1D arrays
    if arr1.ndim == 1:
        plt.figure(figsize=(10, 5))
        plt.plot(arr1, label='Array 1', marker='o')
        plt.plot(arr2, label='Array 2', marker='x')
        plt.plot(weighted_diff, label='Weighted Difference', marker='s', linestyle='--')
        plt.title("Element-wise Comparison")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Element Index")
        plt.ylabel("Value")
    
    plt.show()

    return {
        'array1': arr1,
        'array2': arr2,
        'weights': weights,
        'absolute_differences': abs_diff,
        'weighted_differences': weighted_diff
    }

# Example usage
if __name__ == "__main__":
    # Update these paths to your .npy files
    results = load_and_compare(
        file1='P1_C1_data_iqr_zeroed.npy',
        file2='P1_C1_data_zscore_zeroed.npy',
        weights_file=None
    )