"""Implements private functions for time-domain processing.

Processing is implemented via private functions where that processing
is required other functions and requires data in a specific format.

This avoids reduntant reformatting and input sanatisation.
"""

import numpy as np
import warnings
from .. import _unitchecks


def _format_waveform(waveform, time_unit="ps"):
    # Accepts valid waveforms with varying structure
    # and converts them to a standard format.

    shape = np.shape(waveform)

    # Check that waveform only contains two 1-d datasets.
    if (len(shape) != 2) or (2 not in shape):
        raise ValueError("Waveform could not be processed. "
                         "Please check the format of your data.")

    # Reshape waveforms structured as pairs of values.
    if shape[1] == 2:
        waveform = np.swapaxes(waveform, 0, 1)

    # Ensure the time base is the second dataset.
    # Time dataset is found by checking for continuosly increasing values.
    if min(np.diff(waveform[0])) > 0:
        waveform = waveform[::-1]
    elif min(np.diff(waveform[1])) > 0:
        waveform = waveform
    else:
        raise ValueError("Could not identify a sutiable time axis. "
                         "Please ensure your time values never decrease.")

    waveform[1] = _unitchecks._check_time(waveform[1], time_unit)
    return waveform


def _timebase(time):
    # Calculates the timebase of a time series.
    # Assumes acquistion rate is constant.

    timesteps = np.diff(time)
    timebase = np.mean(timesteps)
    return timebase


def _acq_freq(timebase):
    # Frequency is 1/timebase
    return 1/timebase


def _primary_peak(waveform):
    # Locates the primary peak of a waveform

    field = np.abs(waveform[0])
    time = waveform[1]

    peak_index = np.argmax(field)
    peak_value = field[peak_index]
    peak_time = time[peak_index]

    return (peak_time, peak_value, peak_index)


def _window(ds, centre, n, win_func):
    # Applies the specified window function to a dataset.

    # Ensure window length is even.
    if n % 2 != 0:
        n += 1

    # Generate base window function.
    match win_func:
        case "bartlett":
            window = np.bartlett(n)
        case "blackman":
            window = np.blackman(n)
        case "boxcar":
            window = np.ones(n)
        case "hamming":
            window = np.hamming(n)
        case "hanning":
            window = np.hanning(n)
        case _:
            raise ValueError("Invalid window function.")

    # Calculate required padding and start/stop indexes.
    l_pad = int(max(n/2 - centre, 0))
    r_pad = int(max(n/2 - (len(ds) - centre), 0))
    start = int(centre - n/2) + l_pad
    stop = int(centre + n/2) + l_pad

    if l_pad + r_pad > 0:
        warnings.warn("Window size is larger than dataset. "
                      "Zero padding will be used.")

    ds = np.pad(ds, (l_pad, r_pad))[start:stop]

    return window*ds
