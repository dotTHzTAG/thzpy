"""Provides public public access to THz-TDS transfer functions."""

import numpy as np
from ._transferfunctions import _transform
from .._unitchecks import _check_thickness
from ._transmission import (_uniform_slab,
                            _binary_mixture)
from ..frequencydomain._frequencydomain import (_beer_lambert,
                                                _maxwell_garnett)
from ..frequencydomain._frequencydomain import (_n_complex,
                                                _absorption_coefficient,
                                                _dielectric_constant,
                                                _invert_dielectric_constant,
                                                _all_optical_constants)


def uniform_slab(thickness, sample, baseline,
                 thickness_unit="mm", n_med=1.,
                 min_frequency=0.1, max_frequency=3.5,
                 upsampling=0, peak_snr=1., fit_range=0.2,
                 all_optical_constants=False):
    """Transfer function for a slab of a single medium.

    Calculates the optical constants of a sample composed of a single medium
    from the time-domain waveforms of the sample and the instrument baseline.
    Waveforms are tranformed into the frequency domain using the discrete
    Fourier transform, and the transmission amplitude and phase are computed
    from the DFT output. An infomed phase unwrapping procedure is used
    here to prevent overcorrection of the phase. A transfer function is then
    applied to calculate refractive index and absorption coefficient from the
    phase and amplitude data. These are then used to determine other relevant
    optical constants.

    Parameters
    ----------
    thickness : float
        The thickness of the sample.
    sample : array_like
        The sample waveform. Should take the form of a 2-d array
        containing field amplitude and time data.
    baseline : array_like
        The baseline waveform. Should take the form of a 2-d array
        containing field amplitude and time data.
    thickness_unit : str, optional
        The unit of the thickness value.
        Default is "mm".
    n_med : float or array_like, optional
        The refrective index of the surrounding medium.
        If array_like it's length must match the transformed sample data.
        Default is 1 for air.
    upsampling : int, optional
        Upsampling factor, data is upsampled by 2^upsampling.
        Default is 0.
    peak_snr : float, optional
        The frequency at which the instrument SNR is highest.
        Default is 1 THz.
    fit_range : float, optional
        The frequency span over which to perform fitting for offset
        correction during phase unwrapping.
        Default is 0.2 THz.
    all_optical_constants : bool, optional
        Specifies whether the function should return all optical constants
        or only the complex refractice index.
        Default is False.

    Returns
    -------
    n_complex : array_like
        The complex refractive index of the sample. Formatted as a 2d array.
    optical_constants : dict of array_like, optional
        The optical constants of sample as a dictionary of 1d arrays, includes:
        frequency, phase, transmission amplitude, transmission intensity,
        absorption coefficient, extinction coefficient,
        refractive index, and dielectric constant.
    """
    thickness = _check_thickness(thickness, thickness_unit)

    # Transform to frequency domain.
    amp, phase, freqs = _transform(sample, baseline,
                                   upsampling=upsampling,
                                   peak_snr=peak_snr,
                                   fit_range=fit_range,
                                   min_frequency=min_frequency,
                                   max_frequency=max_frequency)

    # Apply transfer function.
    n, a = _uniform_slab(amp, phase, freqs*1e12, thickness, n_med)

    # Return optical constants.
    if all_optical_constants:
        return _all_optical_constants(n, a, freqs, amp, phase)
    else:
        return np.vstack([_n_complex(n, a, freqs), freqs])


def binary_mixture(sample_thickness, reference_thickness,
                   sample, reference, baseline=None,
                   thickness_unit="mm", n_med=1., n_ref=1.54, a_ref=0,
                   min_frequency=0.1, max_frequency=3.5,
                   upsampling=0, peak_snr=1., fit_range=0.2,
                   effective_medium="maxwell-garnett",
                   all_optical_constants=False):
    """Transfer function for a binary mixture.

    Calculates the optical constants of a sample comprising a slab of a
    well-mixed binary mixture. The sample measurement should be of the mixed
    sample, while the reference measurement should be of one component of the
    misture in isolation.

    Waveforms are tranformed into the frequency
    domain using the discrete Fourier transform, and the transmission
    amplitude and phase are computed from the DFT output. An infomed
    phase unwrapping procedure is used here to prevent overcorrection of the
    phase.

    If all three measurments of sample, reference, and baseline are provided
    then the transfer function for a uniform slab performed for the both the
    sample and reference with respect to baseline. Effective medium theory is
    then used to extract the optical constants of the intrinsic material.

    If only sample and reference waveforms are provided then the refractive
    index of the reference is required as an additional parameter, and a
    different approximate transfer function is used for the sample with respect
    to the reference. Effective medium theory is then used to extract the
    optical constants of the intrinsic material.

    Parameters
    ----------
    sample_thickness : float
        The thickness of the sample.
    reference_thickness : float
        The thickness of the reference.
    sample : array_like
        The sample waveform. Should take the form of a 2-d array
        containing field amplitude and time data.
    reference : array_like
        The reference waveform. Should take the form of a 2-d array
        containing field amplitude and time data.
    baseline : array_like, optional
        The baseline waveform. Should take the form of a 2-d array
        containing field amplitude and time data.
        Default is None.
    thickness_unit : str, optional
        The unit of the thickness values.
        Default is "mm".
    n_med : float or array_like, optional
        The refrective index of the surrounding medium.
        If array_like it's length must match the transformed sample data.
        Default is 1 for air.
    n_ref : float or array_like, optional
        The refrective index of the reference component of the binary mixture.
        If array_like it's length must match the transformed sample data.
        Default is 1.54 for polyethylene.
    upsampling : int, optional
        Upsampling factor, data is upsampled by 2^upsampling.
        Default is 0.
    peak_snr : float, optional
        The frequency at which the instrument SNR is highest.
        Default is 1 THz.
    fit_range : float, optional
        The frequency span over which to perform fitting for offset
        correction during phase unwrapping.
        Default is 0.2 THz.
    effective_medium : str, optional
        Specifies which effective medium theory should be used to determine the
        sample's optical constants.
        The following model are implemented:
        beer-lambert, maxwell-garnett.
        Default is "maxwell-garnett".
    all_optical_constants : bool, optional
        Specifies whether the function should return all optical constants
        or only the complex refractice index.
        Default is False.

    Returns
    -------
    n_complex : array_like
        The complex refractive index of the sample. Formatted as a 2d array.
    optical_constants : dict of array_like, optional
        The optical constants of sample as a dictionary of 1d arrays, includes:
        frequency, phase, transmission amplitude, transmission intensity,
        absorption coefficient, extinction coefficient,
        refractive index, and dielectric constant.
    """
    sample_thickness = _check_thickness(sample_thickness, thickness_unit)
    reference_thickness = _check_thickness(reference_thickness, thickness_unit)

    if baseline is not None:
        # If a baseline is provided treat sample and reference as
        # uniform slabs WRT baseline then apply EMT.

        # Transform to frequency domain.
        amp_mix, phase_mix, freqs_mix = _transform(sample, baseline,
                                                   upsampling=upsampling,
                                                   peak_snr=peak_snr,
                                                   fit_range=fit_range,
                                                   min_frequency=min_frequency,
                                                   max_frequency=max_frequency)
        amp_ref, phase_ref, freqs_ref = _transform(reference, baseline,
                                                   upsampling=upsampling,
                                                   peak_snr=peak_snr,
                                                   fit_range=fit_range,
                                                   min_frequency=min_frequency,
                                                   max_frequency=max_frequency)

        # Apply transfer function.
        n, a = _uniform_slab(amp_mix, phase_mix, freqs_mix*1e12,
                             sample_thickness, n_med)
        n_ref, a_ref = _uniform_slab(amp_ref, phase_ref, freqs_ref*1e12,
                                     reference_thickness, n_med)

        amp = amp_mix/amp_ref
        phase = phase_mix - phase_ref
        freqs = freqs_mix

    else:
        # If no baseline is provided use approximation.

        # Transform to frequency domain.
        amp, phase, freqs = _transform(sample, reference,
                                       upsampling=upsampling,
                                       peak_snr=peak_snr,
                                       fit_range=fit_range,
                                       min_frequency=min_frequency,
                                       max_frequency=max_frequency)

        # Apply transfer function.
        n, a = _binary_mixture(amp, phase, freqs*1e12,
                               sample_thickness, reference_thickness,
                               n_med, n_ref, a_ref)

    # Convert to dielectric constant and apply effective medium theory.
    e_mix = _dielectric_constant(_n_complex(n, a, freqs*1e12))
    e_ref = _dielectric_constant(_n_complex(n_ref, a_ref, freqs*1e12))

    match effective_medium:
        case 'beer-lambert':
            e_sam = _beer_lambert(e_mix, e_ref,
                                  reference_thickness,
                                  sample_thickness-reference_thickness)
        case 'maxwell-garnett':
            e_sam = _maxwell_garnett(e_mix, e_ref,
                                     reference_thickness,
                                     sample_thickness-reference_thickness)
        case _:
            raise ValueError("Invalid effective medium model specified.")

    # Prepare optical constant values for returning.
    n_sam = _invert_dielectric_constant(e_sam)
    n = np.real(n_sam)
    a = _absorption_coefficient(np.imag(n_sam), freqs*1e12)
    
    # Return optical constants.
    if all_optical_constants:
        return _all_optical_constants(n, a, freqs, amp, phase)
    else:
        return np.vstack([_n_complex(n, a, freqs), freqs])
