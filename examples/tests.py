from thzpy import dotthz
from thzpy.timedomain import common_window
from thzpy.transferfunctions import (uniform_slab,
                                     binary_mixture)
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent
path = root.joinpath(r"example_data\Lactose.thz")
wl = 15

# Load a .thz file.
with dotthz.DotthzFile(path, 'r') as file:
    # Get the first measurement in the file.
    names = file.get_measurement_names()
    measurement = file.get_measurement(names[0])

    # Extract the sample and reference datasets.
    sample = measurement.datasets["Sample"]
    reference = measurement.datasets["Reference"]
    baseline = measurement.datasets["Baseline"]

    # Get the thickness metadata.
    metadata = measurement.meta_data.md
    sample_thickness = metadata["Sample Thickness (mm)"]
    reference_thickness = metadata["Reference Thickness (mm)"]

    # Window datasets
    sample, reference, baseline = common_window([sample, reference, baseline],
                                                wl, win_func="hanning")

optical_constants = uniform_slab(reference_thickness,
                                 reference, baseline,
                                 upsampling=3,
                                 all_optical_constants=True)

n_ref_exp = np.mean(np.real(optical_constants["refractive_index"]))

optical_constants_2 = binary_mixture(sample_thickness, reference_thickness,
                                     sample, reference,
                                     upsampling=3,
                                     all_optical_constants=True,
                                     n_ref=n_ref_exp)

optical_constants_3 = binary_mixture(sample_thickness, reference_thickness,
                                     sample, reference, baseline,
                                     upsampling=3,
                                     all_optical_constants=True,
                                     effective_medium="maxwell-garnett")

plt.subplot(2, 2, 1)
plt.plot(sample[1], sample[0], 'r')
plt.plot(reference[1], reference[0] + 0.5, 'g')
plt.plot(baseline[1], baseline[0] + 1, 'b')

plt.subplot(2, 2, 2)
plt.plot(optical_constants["frequency"],
         optical_constants["transmission_amplitude"], 'r')
plt.plot(optical_constants_2["frequency"],
         optical_constants_2["transmission_amplitude"], 'g')
plt.plot(optical_constants_3["frequency"],
         optical_constants_3["transmission_amplitude"], 'b')

plt.subplot(2, 2, 3)
plt.plot(optical_constants["frequency"],
         np.real(optical_constants["refractive_index"]), 'r')
plt.plot(optical_constants_2["frequency"],
         np.real(optical_constants_2["refractive_index"]), 'g')
plt.plot(optical_constants_3["frequency"],
         np.real(optical_constants_3["refractive_index"]), 'b')

plt.subplot(2, 2, 4)
plt.plot(optical_constants["frequency"],
         optical_constants["absorption_coefficient"], 'r')
plt.plot(optical_constants_2["frequency"],
         optical_constants_2["absorption_coefficient"], 'g')
plt.plot(optical_constants_3["frequency"],
         optical_constants_3["absorption_coefficient"], 'b')

plt.show()
print("Done!")
