"""Functions for transforming from time domain to frequency domain."""

import numpy as np
from ..timedomain._timedomain import _timebase


def _transform(sample, reference,
               upsampling=3, peak_snr=1, fit_range=0.2,
               min_frequency=0, max_frequency=5):
    # Using FFT get the transmission and phase amplitude
    # from two THz-TDS waveforms.

    # Calcuate FFT sample points.
    n = (1 << (len(sample[0])-1).bit_length())*(2**upsampling)

    # Perform Fourier transform.
    timebase = _timebase(sample[1])
    freqs = np.fft.fftfreq(n, timebase)
    sample_fd = np.fft.fft(sample[0], n)
    reference_fd = np.fft.fft(reference[0], n)
    amplitude = abs(sample_fd)/abs(reference_fd)

    # Unwrap phases.
    sample_peak = sample[1][np.argmax(sample[0])]
    reference_peak = reference[1][np.argmax(reference[0])]
    phase = _unwrap(sample_fd, reference_fd, sample_peak, reference_peak,
                    freqs, peak_snr, fit_range/2)

    # Remove negative frequencies
    valid_indices = np.where(np.logical_and(freqs >= min_frequency,
                                            freqs <= max_frequency))
    freqs = freqs[valid_indices]
    phase = phase[valid_indices]
    amplitude = amplitude[valid_indices]

    return amplitude, phase, freqs


def _unwrap(sample_fd, reference_fd,
            sample_peak, reference_peak,
            freqs, peak_snr, fit_range):
    # Umwrap phase data using Jepsen's method:
    # https://doi.org/10.1007/s10762-019-00578-0

    # Reduce phase to avoid overcorrection when unwrapping
    k1 = 2*np.pi*sample_peak*freqs
    phase1 = np.angle(sample_fd*(np.e**(-1j*k1)))

    k2 = 2*np.pi*reference_peak*freqs
    phase2 = np.angle(reference_fd*(np.e**(-1j*k2)))
    phase_unwrapped = np.unwrap(phase1 - phase2)

    # Perform a linear fit at a high SNR region to determine phase offset
    left_index = np.argmin(abs(freqs - (peak_snr - fit_range)))
    right_index = np.argmin(abs(freqs - (peak_snr + fit_range)))
    a = np.vstack([freqs[left_index:right_index],
                   np.ones(right_index - left_index)]).T
    _, c = np.linalg.lstsq(a, phase_unwrapped[left_index:right_index])[0]
    offset = 2*np.pi*(round(c/(2*np.pi)))

    return abs(phase_unwrapped + (k1 - k2) - offset)
