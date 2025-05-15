"""Implements public functions for time domain processing.

Where user access to private functions is required they are
exposed here via public functions.
"""

import numpy as np
from .. import _unitchecks
from ._timedomain import (_primary_peak,
                          _format_waveform,
                          _timebase,
                          _acq_freq,
                          _window)


def timebase(waveform, time_unit="ps"):
    """Find the time base of a waveform
    (i.e. the average time step between acquisitions).

    Parameters
    ----------
    waveform : array_like
        The waveform to be windowed. Should take the form of a 2-d array
        containing field amplitude and time data.
    time_unit : str, optional
        The unit of the time values.
        Default is "ps".

    Returns
    -------
    timebase: float
        The average time between acquisitions.
    """
    time = _format_waveform(waveform, time_unit)[1]
    return _timebase(time)


def acq_freq(waveform, time_unit="ps"):
    """Finds the acqusition frequency of a waveform.

    Parameters
    ----------
    waveform : array_like
        The waveform to be windowed. Should take the form of a 2-d array
        containing field amplitude and time data.
    time_unit : str, optional
        The unit of the time values.
        Default is "ps".

    Returns
    -------
    frequency : float
        The acquistion frequency.
    """
    time = _format_waveform(waveform, time_unit)[1]
    return _acq_freq(time)


def primary_peak(waveform, time_unit="ps"):
    """Finds the height and location of the main peak in a waveform.

    Parameters
    ----------
    waveform : array_like
        The waveform to be windowed. Should take the form of a 2-d array
        containing field amplitude and time data.
    time_unit : str, optional
        The unit of the time values, default is "ps".

    Returns
    -------
    tuple of floats
        The time, value, and index of the waveform peak.
    """

    waveform = _format_waveform(waveform, time_unit)
    return _primary_peak(waveform)


def n_effective(sample, ref, thickness,
                ref_thickness=0., n_medium=1.,
                thickness_unit='mm', time_unit="ps"):
    """Calculate the effective refractive index of a sample
    from the phase shift between sample and reference waveforms.

    Parameters
    ----------
    sample : array_like
        The sample waveform. Should take the form of a 2-d array
        containing field amplitude and time data.
    ref : array_like
        The reference waveform. Should take the form of a 2-d array
        containing field amplitude and time data.
    thickness : float
        The thickness of the sample.
    ref_thickness : float, optional
        The thickness of the reference.
        Default is 0.
    n_medium : float, optional
        The refractive index of the medium surrounding the sample/reference.
        Default is 1 for air.
    thickness_unit : str, optional
        The unit of the thickness values.
        Default is "mm".
    time_unit : str, optional
        The unit of the time values.
        Default is "ps".

    Returns
    -------
    n_effective : float
        The effective refractive index of the sample.
    """

    # Reformat waveforms.
    ref = _format_waveform(ref, time_unit)
    sample = _format_waveform(sample, time_unit)

    # Calculate the phase shift between reference and sample.
    ref_peak_time = _primary_peak(ref)[0]
    sample_peak_time = _primary_peak(sample)[0]
    time_delay = sample_peak_time - ref_peak_time
    if time_delay < 0:
        raise ValueError("Negative time delay found. "
                         "Please check the ordering of your datasets.")

    # Calculate sample thickness.
    thickness -= ref_thickness
    if thickness < 0:
        raise ValueError("Negative thickness specified.")
    thickness = _unitchecks._check_thickness(thickness, thickness_unit)

    # Calculate and return the refractive index.
    n_effective = ((299792458*time_delay*1e-12)/thickness) + n_medium
    return n_effective


def window(waveform, half_width, win_func="hanning", time_unit="ps"):
    """Applies a window function to a waveform.

    Parameters
    ----------
    waveform : array_like
        The waveform to be windowed. Should take the form of a 2-d array
        containing field amplitude and time data.
    half_width : float
        The half width of the window
        (i.e how far it should extend each side of the peak).
    win_func : str, optional
        The window function to be applied.
        The following wavefunctions are implemented:
        boxcar, bartlett, blackman, hamming, hanning.
        Default is "hanning".
    time_unit : str, optional
        The unit of the time values.
        Default is "ps".

    Returns
    -------
    windowed_waveform : ndarray
        The input waveform with the specified window function applied.
        The returned waveform has the structure [Field, Time].
    """

    # Format the waveform and extract the field and time data.
    waveform = _format_waveform(waveform, time_unit)
    field = waveform[0]
    time = waveform[1]

    # Calculate window size.
    _, _, peak_index = _primary_peak(waveform)
    dt = _timebase(time)
    n = round(half_width/dt)

    # Apply the window function.
    field = _window(field, peak_index, 2*n, win_func=win_func)

    # Regularise the time axis.
    time = dt*np.arange(-n, n) + time[peak_index]

    return np.array([field, time])


def common_window(waveforms, half_width, win_func="hanning", time_unit="ps"):
    """Applies the same window function to a set of waveforms.
    Padding is automatically applied to preserve the phase shift
    between waveforms and waveforms are re-interpolated on to a
    common time axis.

    Parameters
    ----------
    waveforms : array_like
        The waveforms to be windowed. Should take the form of a 3-d array or a
        list of 2-d arrays containing field amplitude and time data.
    half_width : float
        The half width of the window
        (i.e how far it should extend each side of the peak).
    win_func : str, optional
        The window function to be applied.
        The following wavefunctions are implemented:
        boxcar, bartlett, blackman, hamming, hanning.
        Default is "hanning".
    time_unit : str, optional
        The unit of the time values.
        Default is "ps".

    Returns
    -------
    windowed_waveforms : ndarray
        The input waveforms with the specified window function applied.
        The returned waveforms have the structure
        [[Field 1, Time], [Field2, Time], ...].
    """

    peak_times = []
    windowed_waveforms = []
    output_waveforms = []
    waveforms = [_format_waveform(waveform, time_unit)
                 for waveform in waveforms]

    # Find primary peak of all waveforms
    for waveform in waveforms:
        peak_time, _, _ = _primary_peak(waveform)
        peak_times.append(peak_time)

    # Find the new half width required to fit all waveforms.
    max_delay = max(peak_times) - min(peak_times)
    common_half_width = half_width - max_delay/2

    # Window waveforms
    # TODO: Remove call to public function.
    for waveform in waveforms:
        windowed_waveforms.append(window(waveform,
                                         common_half_width,
                                         win_func=win_func))

    # Get sample waveform index.
    # Interpolation and padding will be done relative to the sample.
    index = np.argmax(peak_times)
    time = windowed_waveforms[index][1]
    dt = _timebase(time)
    n = len(time)
    max_pad = round(max_delay/dt) + 1
    time = np.concatenate([dt*np.arange(-max_pad, 0) + time[0], time])

    for i in range(len(windowed_waveforms)):
        waveform = windowed_waveforms[i]
        field = waveform[0]
        # If necessary interpolate waveforms to a consistent sampling rate.
        if len(field) != n:
            field = np.interp(np.arange(n), np.arange(len(field)), field)

        # Apply padding to preserve alignments.
        right_pad = round((peak_times[index] - peak_times[i])/dt)
        left_pad = max_pad - right_pad
        field = np.pad(field, (left_pad, right_pad))

        output_waveforms.append(np.vstack([field, time]))

    return output_waveforms
