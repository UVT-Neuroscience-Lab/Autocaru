import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = scipy.io.loadmat('P1_post_test.mat')
trig = data['trig']
y = data['y']

# Electrode names mapping (0-based index)
electrode_names = [
    (0, "FC3"), (1, "FCz"), (2, "FC4"),
    (3, "C5"), (4, "C3"), (5, "C1"),
    (6, "Cz"), (7, "C2"), (8, "C4"),
    (9, "C6"), (10, "CP3"), (11, "CP1"),
    (12, "CPz"), (13, "CP2"), (14, "CP4"),
    (15, "Pz")
]

# Create individual plots for each sensor
for sensor_idx, sensor_name in electrode_names:
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Plot trigger signal
    ax1.plot(trig, color='red')
    ax1.set_title(f"Trigger Signal - {sensor_name}")
    ax1.set_ylabel("Trigger State")
    ax1.set_yticks([0, 1])
    ax1.grid(True, alpha=0.3)

    # Plot sensor data
    ax2.plot(y[:, sensor_idx], color='blue')
    ax2.set_title(f"Sensor Data - {sensor_name}")
    ax2.set_xlabel("Time Samples")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)

    # Adjust spacing and save
    plt.tight_layout()
    plt.savefig(f"{sensor_name}_plot.png", dpi=150, bbox_inches='tight')
    plt.close()  # Prevent memory overload

print("All sensor plots saved as PNG files!")