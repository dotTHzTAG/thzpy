"""Transfer functions for transmission geometries."""

import numpy as np
from ..timedomain._timedomain import _timebase


def _uniform_slab(amp, phase, freqs, thickness, n_med=1.,):
    # Calculates the complex refractive index for a homogenous slab.
    # Derivations for transfer function from Leung et al.
    # https://doi.org/10.1007/s10762-025-01070-8

    # Ensure provided refractive index only uses the real component.
    n_med = np.real(n_med)

    n = (299792458*phase)/(2*np.pi*freqs*thickness) + n_med
    a = (2/thickness)*np.log((4*n*n_med)/(amp*((n_med + n)**2)))

    return (n, a)


def _binary_mixture(amp, phase, freqs, t_sam, t_ref,
                    n_med=1., n_ref=1.44, a_ref=0.):
    # Calculates the complex refractive index for a slab
    # composed of two well mixed constituants.
    # Derivations for transfer function from Leung et al.
    # https://doi.org/10.1007/s10762-025-01070-8

    # Ensure provided refractive indices only use the real component.
    n_med = np.real(n_med)
    n_ref = np.real(n_ref)

    n = ((299792458*phase)/(2*np.pi*freqs*t_sam)
         + (t_ref*(n_ref - n_med))/t_sam
         + n_med)
    a = (a_ref*(t_ref/t_sam)
         + (2/t_sam)*np.log((n*((n_med+n_ref)**2))/(amp*n_ref*((n_med+n)**2))))

    return (n, a)


def _amaxd(baseline, n, f, snr):
    # Calculates maximum measurable absorption for a spectrometer and material
    # follwing the method from Jepsen's optics letter
    # https://doi.org/10.1364/OL.30.000029

    # Calculate number of sample points to match input frequency resolution
    fr = (f[1] - f[0])
    timebase = _timebase(baseline[1])
    fft_n = int(1/(timebase*fr))

    # Transform the baseline waveform
    fd = np.fft.fft(baseline[0], fft_n)
    freqs = np.fft.fftfreq(fft_n, timebase)

    # Trim and interpolate data to match input frequency range
    valid_indices = np.where(np.logical_and(freqs >= f[0],
                                            freqs <= f[-1]))
    freqs = freqs[valid_indices]
    fd = fd[valid_indices]
    fd = np.interp(f, freqs, fd)

    # Calculate and smooth the baseline instensity spectrum.
    wl = len(fd)//10
    window = np.ones(wl)/wl
    fd_db = 20*np.log10((fd/np.max(fd)))
    fd_db_smooth = np.convolve(np.pad(fd_db, (wl//2, wl-1-wl//2), mode='edge'),
                               window, mode="valid")

    # Normalise the spectrum to be 1 at the noise floor for a given snr
    cutoff_index = np.argmin(abs(fd_db_smooth + snr))
    fd_norm = fd/fd[cutoff_index]

    # Compute the maximum absorption following Jepsen: 10.1364/OL.30.000029
    amaxd = 2*np.log(fd_norm*((4*n)/((n + 1)**2)))

    return amaxd
