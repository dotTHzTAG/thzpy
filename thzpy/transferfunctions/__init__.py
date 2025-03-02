"""
Transfer Functions (:mod:`thzpy.transferfunctions`)
=====================================

.. currentmodule:: thzpy.transferfunctions

To determine optical constants using terahertz time domain it is necessary
to convert data into the frequency domain using the fourier transform. This
frequency domain data must then be further processed to account for the sample
composition and geometry. This is done with using an equation known as a
transfer function which must be specifically formulated for different cases.

Time-domain data is fourier transformed using the DFT and an informed phase
unwrapping procedure is applied yielding transmission amplitude and phase data
This is then input into a transfer function alongside neccessary information
such as sample thickness to determine the material constants.

Transmission
-------------

   .. toctree::

    homogenous_slab     Transfer function for a slab of a single medium.
    binary_mixture      Transfer function for a slab of a binary mixture.
    bilayer             TO BE IMPLEMENTED
    cuvette             TO BE IMPLEMENTED


Reflection
-------------

   .. toctree::

   TO BE IMPLEMENTED


Transflection
-------------

   .. toctree::

   TO BE IMPLEMENTED

"""

from .transferfunctions import (uniform_slab,
                                binary_mixture)

__all__ = [uniform_slab,
           binary_mixture]
