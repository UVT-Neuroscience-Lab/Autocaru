import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

data = scipy.io.loadmat('P1_post_test.mat')
trig = data['trig']
y = data['y']

plt.figure(figsize=(12, 2))
plt.plot(trig)
plt.title("Trigger Signal")
plt.xlabel("Time Samples")
plt.ylabel("Trigger State")
plt.yticks([0, 1])
plt.show()

plt.figure(figsize=(12, 6))
for i in range(y.shape[1]):
    plt.plot(y[:, i], label=f'Sensor {i+1}', alpha=0.7)
plt.title("Sensor Signals")
plt.xlabel("Time Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Calculate standard deviation per sensor
sensor_std = np.std(y, axis=0)
print("Sensor activity (STD):", sensor_std)

# Identify active sensors (arbitrary threshold)
active_threshold = 1000  # Adjust based on your data
active_sensors = np.where(sensor_std > active_threshold)[0]
print("Active sensors:", active_sensors)