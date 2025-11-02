"""
An example of data processing to determine the dynamic range of a measurment.

Demonstrates how to extract absorption coefficient data from a set of
baseline and sample waveforms, then determining the associated dynamic
range for the combination of material acquisition parameters. The absorption
spectrum of PLA measured from three different thicknesses is used for general
demonstration.

Data acquired on a Menlo TeraSmart by the Terahertz Applications Group
at Cambridge University.
"""

from thzpy.dotthz import DotthzFile
from thzpy.timedomain import common_window
from thzpy.transferfunctions import uniform_slab
from thzpy.frequencydomain import find_dynamic_range
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#############
# Load Data #
#############

root = Path(__file__).parent
path = root.joinpath("example_data", "PLA_DR.thz")  # OS-independent joining

measurements = {}
thicknesses = {}

# Extract the PLA data.
with DotthzFile(path, 'r') as file:
    # Get the measurement names.
    names = file.get_measurement_names()

    for name in names:
        measurement = file[name]

        # Extract the sample, reference and baseline datasets.
        # The baseline waveform is required later for computing dynamic range
        # so it it is preserved outside the file context using np.array().
        sample = measurement.datasets["Sample"]
        reference = measurement.datasets["Reference"]
        baseline = np.array(measurement.datasets["Baseline"])

        # Get the thickness metadata and save it outside the file context.
        metadata = measurement.metadata
        sample_thickness = metadata["Sample_Thickness(mm)"]
        thicknesses[name] = sample_thickness

        # Apply a window function to all datasets with a half-width of 15 ps.
        sample, reference, baseline = common_window([sample,
                                                     reference,
                                                     baseline],
                                                    half_width=20,
                                                    win_func="hanning")

        # Compute the optical properties of the sample.
        measurements[name] = uniform_slab(sample_thickness,
                                          sample, reference,
                                          upsampling=3,
                                          min_frequency=0.2, max_frequency=8,
                                          all_optical_constants=True)

###################
# Data Processing #
###################

# The refractive index of the material is required as for calculating the
# dynamic range, it may be a single effective value or a spectrum. If it
# is a spectrum it must also match the array of frequencies.
refractive_index = np.real(measurements["PLA_0.91mm"]["refractive_index"])
frequency = measurements["PLA_0.91mm"]["frequency"]


# Compute the maximum measurable absorption coefficient at each frequency.
amax = find_dynamic_range(baseline, frequency, refractive_index,
                          thickness=thicknesses["PLA_0.91mm"],
                          snr=90, mode="amax")

# Alternatively when you have multiple samples of the same material compute
# amaxd. When divided by sample thickness this becomes the maximum absorption
# coefficient. This is avoids redundent computation.
amaxd = find_dynamic_range(baseline, frequency, refractive_index,
                           snr=90, mode="amaxd")

# For simplicity boundary mode may instead be used. In boundary mode rather
# than a spectrum of amax a list will be returned containing pairs of
# start/stop frequencies for regions outside the dynamic range. Sample
# absorption coefficient is required as an additional parameter.
boundaries = {}
for k, v in measurements.items():
    a = v["absorption_coefficient"]
    boundary = find_dynamic_range(baseline, frequency, refractive_index,
                                  thickness=thicknesses[k],
                                  absorption_coefficient=a,
                                  snr=90, mode="boundaries")
    boundaries[k] = boundary


####################
# Plotting Figures #
####################

plt.figure(figsize=(10, 5), dpi=80)

colours = ["r", "g", "b"]

for k, v in measurements.items():
    colour = colours.pop()

    # Plot boundaries
    for r in boundaries[k]:
        plt.axvspan(r[0], r[1], color=colour, alpha=0.1)

    # Plot amax
    plt.plot(amaxd[1], amaxd[0]/(thicknesses[k]*0.1), 'k--')

    # Plot absorption coefficient
    plt.plot(v["frequency"], v["absorption_coefficient"],
             color=colour, label=k)

plt.xlim((0.2, 8))
plt.legend()
plt.title("Menlo PLA Dynamic Range")
plt.xlabel("Frequency (THz)")
plt.ylabel("Absorption Coefficient (cm-1)")
plt.show()
print("Done!")
