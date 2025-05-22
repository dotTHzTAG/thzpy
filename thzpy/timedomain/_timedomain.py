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
    return 1 / timebase


def _primary_peak(waveform):
    # Locates the primary peak of a waveform

    field = np.abs(waveform[0])
    time = waveform[1]

    peak_index = np.argmax(field)
    peak_value = field[peak_index]
    peak_time = time[peak_index]

    return (peak_time, peak_value, peak_index)


def _window(ds, centre, n, win_func, time=None, adapted_blackman_params=None):
    """Apply a windowing function to a dataset.

    Parameters
    ----------
    ds : array_like
        The dataset to apply windowing to
    centre : int
        The index of the peak in the dataset
    n : int
        The size of the window
    win_func : str
        The window function to use
    time : array_like, optional
        Time data needed for the toptica window
    adapted blackman : tuple, optional
        (start, end) parameters for adapted blackman window

    Returns
    -------
    array_like
        Windowed dataset with same length as input window (n)
    """
    # Ensure window length is even
    if n % 2 != 0:
        n += 1

    # Handle toptica window type
    if win_func == "adapted blackman":
        if time is None:
            raise ValueError("Time data must be provided for adapted blackman window")

        # Use default params if none provided
        if adapted_blackman_params is None:
            start, end = 1, 7
        else:
            start, end = adapted_blackman_params

        # Calculate required padding and start/stop indexes
        l_pad = int(max(n / 2 - centre, 0))
        r_pad = int(max(n / 2 - (len(ds) - centre), 0))
        start_idx = int(centre - n / 2) + l_pad
        stop_idx = int(centre + n / 2) + l_pad

        if l_pad + r_pad > 0:
            import warnings
            warnings.warn("Window size is larger than dataset. Zero padding will be used.")

        # Create padded arrays
        padded_ds = np.pad(ds, (l_pad, r_pad))

        # Extract window segment
        windowed_ds = padded_ds[start_idx:stop_idx]

        # For applying the toptica window, we need the corresponding time values
        if len(time) == len(ds):
            dt = time[1] - time[0]  # Assuming uniform time spacing
            padded_time = np.pad(time, (l_pad, r_pad),
                                 mode='linear_ramp',
                                 end_values=(time[0] - l_pad * dt, time[-1] + r_pad * dt))
            window_time = padded_time[start_idx:stop_idx]

            # Apply blackman window to start and end regions
            def blackman_func(n, M):
                return 0.42 - 0.5 * np.cos(2 * np.pi * n / M) + 0.08 * np.cos(4 * np.pi * n / M)

            window = np.ones_like(windowed_ds)

            # Apply to beginning
            start_region = window_time <= (window_time[0] + start)
            if np.any(start_region):
                a_time = window_time[start_region]
                if len(a_time) > 1:
                    a = blackman_func(a_time - a_time[0], 2 * (a_time[-1] - a_time[0]))
                    window[start_region] = a

            # Apply to end
            end_region = window_time >= (window_time[-1] - end)
            if np.any(end_region):
                b_time = window_time[end_region]
                if len(b_time) > 1:
                    b = blackman_func(b_time + b_time[-1] - b_time[0] - b_time[0],
                                      2 * (b_time[-1] - b_time[0]))
                    window[end_region] = b
            return window * windowed_ds
        else:
            raise ValueError("Time array length must match dataset length")

    # Standard window functions
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
            raise ValueError(f"Invalid window function: {win_func}")

    # Calculate required padding and start/stop indexes
    l_pad = int(max(n / 2 - centre, 0))
    r_pad = int(max(n / 2 - (len(ds) - centre), 0))
    start = int(centre - n / 2) + l_pad
    stop = int(centre + n / 2) + l_pad

    if l_pad + r_pad > 0:
        import warnings
        warnings.warn("Window size is larger than dataset. Zero padding will be used.")

    padded_ds = np.pad(ds, (l_pad, r_pad))
    windowed_ds = window * padded_ds[start:stop]

    return windowed_ds
