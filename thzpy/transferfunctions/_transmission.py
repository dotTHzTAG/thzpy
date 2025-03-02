"""Transfer functions for transmission geometries."""

import numpy as np


def _uniform_slab(amp, phase, freqs, thickness, n_med=1.,):
    # Calculates the complex refractive index for a homogenous slab.
    # TODO: Insert reference to Chi-Ki's paper.

    n = (299792458*phase)/(2*np.pi*freqs*thickness) + n_med
    a = (2/thickness)*np.log((4*n*n_med)/(amp*((n_med + n)**2)))

    return (n, a*1e-2)


def _binary_mixture(amp, phase, freqs, t_sam, t_ref,
                    n_med=1., n_ref=1.44, a_ref=0.):
    # Calculates the complex refractive index for a slab
    # composed of two well mixed constituants.
    # TODO: Insert reference to Chi-Ki's paper.

    n = ((299792458*phase)/(2*np.pi*freqs*t_sam)
         + (t_ref*(n_ref - n_med))/t_sam
         + n_med)
    a = (a_ref*(t_ref/t_sam)
         + (2/t_sam)*np.log((n*((n_med+n_ref)**2))/(amp*n_ref*((n_med+n)**2))))

    return (n, a*1e-2)
