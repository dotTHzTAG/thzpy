"""Implements private functions for frequency-domain processing."""

import numpy as np


##############################
# Optical constant calulations
##############################

def _n_complex(n, a, f):
    k = _extinction_coefficient(a, f)
    return n + 1j*k 

def _extinction_coefficient(a, f):
    w = 2*np.pi*f
    return (299792458*a)/(2*w)


def _absorption_coefficient(k, f):
    return (4*np.pi*f*k)/299792458


def _dielectric_constant(n_complex):
    return n_complex**2


def _invert_dielectric_constant(e_complex):
    return np.sqrt(e_complex)


def _all_optical_constants(n, a, f, t, p):
    # Produce a dictionary of all optical constants.

    # convert units to SI
    # define new parameters so that output parameters would not be affected    
    f_hz = f*1e12 
    a_m = a*100
    
    k = _extinction_coefficient(a_m, f_hz) 
    n_complex = _n_complex(n, a_m, f_hz)
    e_complex = _dielectric_constant(n_complex)

    optical_constants = {"frequency": f,
                         "phase": p,
                         "transmission_amplitude": t,
                         "transmission_intensity": t**2,
                         "absorption_coefficient": a,
                         "extinction_coefficient": k,
                         "refractive_index": n_complex,
                         "dielectric_constant": e_complex}

    return optical_constants

#########################
# Effective medium theory
#########################

def _beer_lambert(e_eff, e_m, v_m, v_i):
    # Computes the dielectric spectrum of the unknown component
    # of a binary mixture using the Beer-Lambert model.

    f_m = v_m/(v_m + v_i)
    f_i = v_i/(v_m + v_i)
    return (e_eff - f_m*e_m)/f_i


def _maxwell_garnett(e_eff, e_m, v_m, v_i):
    # Computes the dielectric spectrum of the unknown component
    # of a binary mixture using the Maxwell Garnett formula.

    f_i = v_i/(v_m + v_i)
    k = f_i*(e_eff + 2*e_m)
    return e_m*((k + 2*e_eff - 2*e_m)/(k + e_m - e_eff))
