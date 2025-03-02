"""
Time Domain (:mod:`thzpy.timedomain`)
=====================================

.. currentmodule:: thzpy.timedomain

Terahertz data often needs processing in the time domain,
either to directly extract temporal information, or as pre-processing
before conversion to the frequency domain.

This module provides time-domain processing functions in four categories:
Parameter extraction, windowing, peak finding, and etalon correction.

Parameter Extraction
-------------

   .. toctree::

   timebase         Calculates the time base of a waveform.
   acq_freq         Calculates the acquisition frequency of a waveform.
   n_effective      Calculates the effective refractive index of a sample.

Windowing
-------------

   .. toctree::

   window           Applies a window funcion to a waveform.
   common_window    Applies one window function to many waveforms while preserving phase offsets.

Peak Finding
-------------

   .. toctree::

   primary_peak     Finds the time, height, and index of the primary peak of a waveform.
   find_peaks       TO BE IMPLEMENTED

Etalon Correction
-------------

   .. toctree::

   TO BE IMPLEMENTED

"""

from .timedomain import (timebase,
                         primary_peak,
                         n_effective,
                         window,
                         common_window)

__all__ = [
    timebase,
    primary_peak,
    n_effective,
    window,
    common_window
]
