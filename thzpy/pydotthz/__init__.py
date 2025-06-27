"""
DotTHz (:mod:`thzpy.pydotthz`)
=====================================

.. currentmodule:: thzpy.pydotthz

This module provides interfacing with the .thz file format. It acts as
a built in interface to the pydotthz package.

See https://github.com/dotTHzTAG/pydotthz for more details.

dotTHz
-------------

   .. toctree::

   DotthzFile           File class for the .thz format, holding THz time-domain spectroscopy data. Simple wrapper around HDF5 for Python. Fully memory mapped.
   DotthzMetaData       Optional data class holding metadata for measurements in the .thz file format.

"""

from pydotthz import (DotthzFile,
                      DotthzMetaData)

__all__ = [DotthzFile,
           DotthzMetaData]
