"""Implements public functions for frequency-domain processing.

Where user access to private functions is required they are
exposed here via public functions.
"""

import numpy as np
from ..transferfunctions._transmission import _amaxd
from .._unitchecks import _check_thickness


def find_dynamic_range(baseline, frequencies,
                       refractive_index,
                       snr=70,
                       absorption_coefficient=None,
                       thickness=None,
                       mode="boundaries",
                       thickness_unit="mm"):
    """Find the dynamic range for a given material spectrum and spectrometer.

    Calculates the maximum measurable absorption the can be measured for a
    given spectrometer configuration and material refractive index. This is
    done by transforming the waveform of the spectrometer baseline and scaling
    it to the noise floor of the instrument. Depending on the specified mode
    the function it will return the maximum absorption, the maximum absorption
    coefficient, or a list containing start/stop frequencies corresponding to
    regions below the dynamic range (defined as areas where absorption
    coefficient is more than 80% of the maximum measurable value.)

    Parameters
    ----------
    baseline : array_like
        The baseline waveform. Should take the form of a 2-d array
        containing field amplitude and time data.
    frequencies : array_like
        The frequencies
    refractive_index : array_like
        The refractive index of the sample of interest.
        Must align with the value of frequencies provided.
    snr : float, optional
        The signal-to-noise ratio of the baseline in decibels.
        Default is 70.
    absorption_coefficient : array_like, optional
        The absorption coefficient to check for DR boundaries.
        Must align with the value of frequencies provided.
    thickness : float, optional
        The thickness of the sample.
    mode : str, optional
        Output mode for the function.
        Dfault "boundaries"
    thickness_unit : str, optional
        The unit of the thickness value.
        Default is "mm".

    Returns
    -------
    boundaries : list of tuples
        A list containing pairs of start/stop frequencies for regions
        outside the dynamic range.
    amax : array_like
        The maximum measurable absorption. Formatted as a 2d array.
    amaxd : array_like
        The maximum measurable coefficient. Formatted as a 2d array.
    """

    # Convert thickness to cm
    if thickness:
        thickness = _check_thickness(thickness, thickness_unit)*100

    amaxd = _amaxd(baseline, np.real(refractive_index), frequencies, snr)

    match mode:
        case "boundaries":
            if absorption_coefficient is None:
                raise ValueError("Absorption coefficient is None")
            if thickness is None:
                raise ValueError("Thickness is None")

            # Find the boundaries between regions above/below the DR
            amax = amaxd/thickness
            regions = np.less_equal(absorption_coefficient, 0.8*amax)
            boundaries = np.where(regions[:-1] != regions[1:])[0] + 1

            # Make sure the boundaries include the spectrum start/end
            boundaries = np.insert(boundaries, 0, 0)
            length = len(regions) - 1
            if boundaries[-1] != length:
                boundaries = np.append(boundaries, length)

            # Create a list of start/stop frequencies for areas below the DR
            below_dr = []
            for i in range(len(boundaries) - 1):
                start_index = boundaries[i]
                stop_index = boundaries[i + 1]
                if not regions[start_index]:
                    below_dr.append((frequencies[start_index],
                                     frequencies[stop_index]))
            return below_dr

        case "amax":
            if thickness is None:
                raise ValueError("Thickness is None")
            return np.vstack((amaxd/thickness, frequencies))

        case "amaxd":
            return np.vstack((amaxd, frequencies))
