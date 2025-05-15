"""
An example of data processing using data loaded from a .thz file.

Demonstrates how to load a file and get measurement data and metadata
from it, how to apply a window function to a group of datasets,
and how to calculate optical constants using two (sample and reference)
or three (sample, reference, and baseline) datasets.

Data acquired on a Menlo TeraSmart by the Terahertz Applications Group
at Cambridge University.
"""

from thzpy.pydotthz import DotthzFile
from thzpy.timedomain import common_window
from thzpy.transferfunctions import (uniform_slab,
                                     binary_mixture)
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#############
# Load Data #
#############

root = Path(__file__).parent
path = root.joinpath("example_data", "Lactose.thz")  # OS-independent joining

with DotthzFile(path, 'r') as file:
    # Get the first measurement in the file.
    names = file.get_measurement_names()
    measurement = file[names[0]]

    # Extract the sample and reference datasets.
    # if the computation is outside of the file context, we need to copy the data using `np.array()` otherwise the pointer only
    # lives as long as the file is opened (only inside this context here)
    sample = np.array(measurement.datasets["Sample"])
    reference = np.array(measurement.datasets["Reference"])
    baseline = np.array(measurement.datasets["Baseline"])

    # Get the thickness metadata.
    metadata = measurement.meta_data
    sample_thickness = metadata["Sample Thickness (mm)"]
    reference_thickness = metadata["Reference Thickness (mm)"]

###################
# Data Processing #
###################

# Apply a window function to all datasets with a half-width of 15 ps.
sample, reference, baseline = common_window([sample, reference, baseline],
                                            half_width=15, win_func="hanning")

# Calculate buffer material properties.
buffer = uniform_slab(reference_thickness,
                      reference, baseline,
                      upsampling=3, min_frequency=0.2, max_frequency=3,
                      all_optical_constants=True)

# Using the two measurement approximation.
lactose_2 = binary_mixture(sample_thickness, reference_thickness,
                           sample, reference,
                           upsampling=3, min_frequency=0.2, max_frequency=3,
                           all_optical_constants=True,
                           n_ref=1.54)

# Using all three measurements.
lactose_3 = binary_mixture(sample_thickness, reference_thickness,
                           sample, reference, baseline,
                           upsampling=3, min_frequency=0.2, max_frequency=3,
                           all_optical_constants=True,
                           effective_medium="maxwell-garnett")

####################
# Plotting Figures #
####################

plt.figure(figsize=(6, 8), dpi=80)
plt.suptitle("Optical Constants of Lactose")
plt.subplot(2, 1, 1)
plt.xlabel("Frequency (THz)")
plt.ylabel("Refractive Index")
plt.plot(buffer["frequency"],
         np.real(buffer["refractive_index"]),
         'r', label="Polyethylene")
plt.plot(lactose_2["frequency"],
         np.real(lactose_2["refractive_index"]),
         'g', label="Lactose (no baseline)")
plt.plot(lactose_3["frequency"],
         np.real(lactose_3["refractive_index"]),
         'b', label="Lactose (baseline)")
plt.legend(loc="lower right")

plt.subplot(2, 1, 2)
plt.xlabel("Frequency (THz)")
plt.ylabel("Absorption Coefficient ($cm^{-1}$)")
plt.plot(buffer["frequency"],
         buffer["absorption_coefficient"],
         'r', label="Polyethylene")
plt.plot(lactose_2["frequency"],
         lactose_2["absorption_coefficient"],
         'g', label="Lactose (no baseline)")
plt.plot(lactose_3["frequency"],
         lactose_3["absorption_coefficient"],
         'b', label="Lactose (baseline)")
plt.legend(loc="lower right")

plt.show()
print("Done!")
