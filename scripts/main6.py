import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
from pathlib import Path
import os

def zero_outliers_iqr(data, iqr_multiplier=1.5):
    """Set outliers to zero using IQR method"""
    flat = data.flatten()
    q1, q3 = np.quantile(flat, [0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - (iqr_multiplier * iqr)
    upper = q3 + (iqr_multiplier * iqr)
    return np.where((data >= lower) & (data <= upper), data, 0)

def zero_outliers_zscore(data, z_threshold=3):
    """Set outliers to zero using Z-score method"""
    flat = data.flatten()
    if np.std(flat) == 0:  # Handle uniform values
        return data
    z_scores = np.abs(stats.zscore(flat))
    z_scores = z_scores.reshape(data.shape)
    return np.where(z_scores < z_threshold, data, 0)

def process_data(y_data, base_name, output_dir):
    """Process y data and create outputs"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply both zeroing methods
    iqr_zeroed = zero_outliers_iqr(y_data)
    zscore_zeroed = zero_outliers_zscore(y_data)

    # Create visualizations
    create_comparison_plots(y_data, iqr_zeroed, zscore_zeroed, output_dir, base_name)

    # Save zeroed data
    np.save(output_dir / f"{base_name}_iqr_zeroed.npy", iqr_zeroed)
    np.save(output_dir / f"{base_name}_zscore_zeroed.npy", zscore_zeroed)

    return {
        'total_points': y_data.size,
        'iqr_zeros': np.sum(iqr_zeroed == 0),
        'zscore_zeros': np.sum(zscore_zeroed == 0)
    }

def create_comparison_plots(original, iqr_zeroed, zscore_zeroed, output_dir, base_name):
    """Generate comparison plots for y dataset"""
    # Flatten data for plotting
    orig_flat = original.flatten()
    iqr_flat = iqr_zeroed.flatten()
    zscore_flat = zscore_zeroed.flatten()

    # IQR comparison plot
    plt.figure(figsize=(15, 8))
    
    # Original vs IQR Zeroed
    plt.subplot(2, 2, 1)
    plt.title('Original Data (y)')
    plt.plot(orig_flat, 'b-', alpha=0.5)
    plt.subplot(2, 2, 2)
    plt.title('IQR Zeroed Data')
    plt.plot(iqr_flat, 'g-', alpha=0.7)
    
    # Distribution comparison
    plt.subplot(2, 2, 3)
    plt.hist(orig_flat, bins=50, color='blue', alpha=0.7, label='Original')
    plt.hist(iqr_flat[iqr_flat != 0], bins=50, color='green', alpha=0.7, label='IQR Valid')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.hist(orig_flat, bins=50, color='blue', alpha=0.7, label='Original')
    plt.hist(zscore_flat[zscore_flat != 0], bins=50, color='red', alpha=0.5, label='Z-score Valid')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{base_name}_comparison_iqr.png', dpi=150)
    plt.close()
    
    # Z-score comparison plot
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 2, 1)
    plt.title('Original vs Z-score Zeroed')
    plt.plot(orig_flat, 'b-', alpha=0.3, label='Original')
    plt.plot(zscore_flat, 'r-', alpha=0.5, label='Z-score Zeroed')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title('Zeroed Points Comparison')
    plt.bar(['IQR Zeroed', 'Z-score Zeroed'], 
            [np.sum(iqr_flat == 0), np.sum(zscore_flat == 0)],
            color=['green', 'red'])
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{base_name}_comparison_zscore.png', dpi=150)
    plt.close()

def process_file(input_path, output_root):
    """Process a single file, handling trig and y variables"""
    results = []
    file_stem = Path(input_path).stem
    output_dir = Path(output_root) / f"filtered_{file_stem}"
    
    try:
        data = np.load(input_path, allow_pickle=True)
        
        # Handle different data formats
        if isinstance(data, np.lib.npyio.NpzFile):
            data_dict = dict(data)
        elif isinstance(data, np.ndarray):
            if data.dtype.names is not None and 'trig' in data.dtype.names:
                data_dict = {'y': data['y'], 'trig': data['trig']}
            else:
                data_dict = {'y': data}
        else:
            data_dict = data.item() if data.size == 1 else {'y': data}

        if 'trig' in data_dict:
            print(f"Ignoring trig variable in {input_path}")
            
        if 'y' not in data_dict:
            raise ValueError(f"No 'y' variable found in {input_path}")
            
        y_data = data_dict['y']
        res = process_data(y_data, file_stem, output_dir)
        results.append((file_stem, res))
            
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
    
    return results

def main(input_dir, output_root):
    """Main processing function"""
    all_results = []
    
    # Process all .npy files in input directory
    for file_path in Path(input_dir).glob('*.npy'):
        if file_path.is_file():
            print(f"\nProcessing {file_path.name}...")
            file_results = process_file(file_path, output_root)
            all_results.extend(file_results)
    
    # Print summary
    print("\nProcessing Summary:")
    for fname, res in all_results:
        print(f"\nFile: {fname}")
        print(f"Total data points: {res['total_points']}")
        print(f"IQR zeroed points: {res['iqr_zeros']} ({res['iqr_zeros']/res['total_points']:.2%})")
        print(f"Z-score zeroed points: {res['zscore_zeros']} ({res['zscore_zeros']/res['total_points']:.2%})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process .npy files with trig/y variables')
    parser.add_argument('input_dir', help='Directory containing .npy files')
    args = parser.parse_args()
    
    data_dir = os.path.join(args.input_dir, 'filtered_output')  # Separate directory for data files

    # Create root output directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    main(args.input_dir, data_dir)