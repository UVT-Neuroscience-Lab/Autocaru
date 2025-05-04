import argparse
import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Electrode names mapping (0-based index)
    electrode_names = [
        (0, "FC3"), (1, "FCz"), (2, "FC4"),
        (3, "C5"), (4, "C3"), (5, "C1"),
        (6, "Cz"), (7, "C2"), (8, "C4"),
        (9, "C6"), (10, "CP3"), (11, "CP1"),
        (12, "CPz"), (13, "CP2"), (14, "CP4"),
        (15, "Pz")
    ]
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process and plot EEG sensor data')
    parser.add_argument('input_file', type=str, help='Path to .mat input file')
    args = parser.parse_args()

    output_dir = args.input_file.split('.')[0]
    data_dir = os.path.join(output_dir, 'data')  # Separate directory for data files
    
    # Create output directories
    os.makedirs(data_dir, exist_ok=True)

    # Load data
    data = scipy.io.loadmat(args.input_file)
    trig = data['trig']
    y = data['y']

    # Save full dataset in numpy format
    np.savez(os.path.join(data_dir, 'full_data.npz'), trig=trig, y=y)
    
    # Create pandas DataFrame with time index
    df = pd.DataFrame(y, columns=[name for _, name in electrode_names])
    df['trigger'] = trig
    df.to_parquet(os.path.join(data_dir, 'full_data.parquet'))
    df.to_csv(os.path.join(data_dir, 'full_data.csv'))

    # Electrode names mapping (0-based index)
    electrode_names = [
        (0, "FC3"), (1, "FCz"), (2, "FC4"),
        (3, "C5"), (4, "C3"), (5, "C1"),
        (6, "Cz"), (7, "C2"), (8, "C4"),
        (9, "C6"), (10, "CP3"), (11, "CP1"),
        (12, "CPz"), (13, "CP2"), (14, "CP4"),
        (15, "Pz")
    ]

    # Create individual plots and data files
    for sensor_idx, sensor_name in electrode_names:
        # Plotting code (same as before)...
        # ... [your existing plotting code here] ...

        # Save sensor-specific data
        sensor_data = y[:, sensor_idx]
        
        # NumPy format
        np.save(
            os.path.join(data_dir, f"{sensor_name}_data.npy"),
            np.column_stack((trig, sensor_data))
        )
        
        # Pandas format
        sensor_df = pd.DataFrame({
            'time': np.arange(len(sensor_data)),
            'trigger': trig.flatten(),
            sensor_name: sensor_data
        })
        sensor_df.to_parquet(os.path.join(data_dir, f"{sensor_name}_data.parquet"))
        sensor_df.to_csv(os.path.join(data_dir, f"{sensor_name}_data.csv"))

    print(f"All data saved to: {os.path.abspath(data_dir)}")
    print("Formats available:")
    print("- NPY/NPZ (NumPy binary)")
    print("- CSV (Excel-compatible)")
    print("- Parquet (compressed columnar format)")

if __name__ == "__main__":
    main()