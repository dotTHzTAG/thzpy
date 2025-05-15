"""
DotTHz (:mod:`thzpy.dotthz`)
=====================================

.. currentmodule:: thzpy.dotthz

This module provides interfacing with the .thz file format. It acts as
a built in interface to the pydotthz package.

See https://github.com/dotTHzTAG/pydotthz for more details.

dotTHz
-------------

   .. toctree::

   DotthzFile           File class for the .thz format, holding THz time-domain spectroscopy data.
   DotthzMeasurement    Data class for terahertz time-domain spectroscopy measurements.
   DotthzMetaData       Data class holding metadata for measurements in the .thz file format.

"""

from pydotthz import (DotthzFile,
                      DotthzMeasurement,
                      DotthzMetaData)

__all__ = [DotthzFile,
           DotthzMeasurement,
           DotthzMetaData]
