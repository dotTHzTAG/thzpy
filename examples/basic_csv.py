"""
An example of data processing using data loaded from a .thz file.

Demonstrates how to load measurement data from a csv, apply a common
window function to the data, and calculate optical constant using two
datasets (sample and reference).

Sample data acquired on a Toptica TeraFlash by the Korter Research Group
at Syracuse University.
"""

import csv
from thzpy.timedomain import common_window
from thzpy.transferfunctions import binary_mixture
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#############
# Load Data #
#############

root = Path(__file__).parent
path_reference = root.joinpath(r"example_data\ptfe.csv")
path_sample = root.joinpath(r"example_data\lactose.csv")

# Read data from files as floats.
with open(path_reference, 'r', newline='') as csvfile:
    time = []
    amp = []
    reference_data = list(csv.reader(csvfile))
    reference_data.remove(['Time_abs/ps', ' Signal/nA'])
    for col in reference_data:
        time.append(float(col[0]))
        amp.append(float(col[1]))
    reference = [amp, time]

with open(path_sample, 'r', newline='') as csvfile:
    time = []
    amp = []
    sample_data = list(csv.reader(csvfile))
    sample_data.remove(['Time_abs/ps', ' Signal/nA'])
    for col in sample_data:
        time.append(float(col[0]))
        amp.append(float(col[1]))
    sample = [amp, time]


###################
# Data Processing #
###################

# Apply a window function to all datasets with a half-width of 25 ps.
sample, reference = common_window([sample, reference],
                                  half_width=25, win_func="hanning")

# Metadata is not included in the files so must be specified seperately.
sample_thickness = 1.26
reference_thickness = 1.

# Using the two measurement approximation.
lactose = binary_mixture(sample_thickness, reference_thickness,
                         sample, reference,
                         upsampling=3, min_frequency=0.2, max_frequency=3,
                         all_optical_constants=True,
                         n_ref=1.43)


####################
# Plotting Figures #
####################

plt.figure(figsize=(6, 8), dpi=80)
plt.suptitle("Optical Constants of Lactose")
plt.subplot(2, 1, 1)
plt.xlabel("Frequency (THz)")
plt.ylabel("Refractive Index")
plt.plot(lactose["frequency"],
         np.real(lactose["refractive_index"]),
         'b', label="Lactose")
plt.legend(loc="lower right")

plt.subplot(2, 1, 2)
plt.xlabel("Frequency (THz)")
plt.ylabel("Absorption Coefficient ($cm^{-1}$)")
plt.plot(lactose["frequency"],
         lactose["absorption_coefficient"],
         'b', label="Lactose")
plt.legend(loc="lower right")

plt.show()
print("Done!")
