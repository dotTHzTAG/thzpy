# THzPy

THzPy is a scientific computing library for processing terahertz time-domain spectroscopy data.

The aim of this library is simplify and speed up the analysis of THz-TDS data, as well as promote the standardisation
of processing methods between research groups and industry.

Development of this package is ongoing, we are in the process of adding features and support for a wider range of
experimental setups and processes.
If you encounter any bugs or there are any specific features you would like to see added please get in touch.

## Features

- Windowing and filtering.
- Extraction of optical constants.
- Spectral feature extraction. (UPCOMING)
- Support for the DotTHz file format. (see `examples/basic_thz.py` for a possible application)

## Usage

The package can be installed via pip:

```shell
pip install thzpy
```

The package then presents various functions to use like so:

```python
from thzpy.timedomain import common_window
from thzpy.transferfunctions import uniform_slab

# Apply the same hanning window to all datasets with a half-width of 15 ps.
reference, baseline = common_window([reference, baseline],
                                    half_width=15, win_func="hanning")

# Calculate buffer material properties.
buffer = uniform_slab(reference_thickness,
                      reference, baseline,
                      upsampling=3, min_frequency=0.2, max_frequency=3,
                      all_optical_constants=True)
```             

Please see the examples folder for full demonstrations of how to use the package.

## Documentation

[GitHub](https://github.com/dotTHzTAG/thzpy)

## Authors

- [@JasperWB](https://www.github.com/JasperWB) - primary development
- [@hacknus](https://github.com/hacknus) - .thz file handling
- [@dotTHzTAG](https://www.github.com/dotTHzTAG) - various contributions

## Feedback

If you have any feedback, please reach out to us at jnw35@cam.ac.uk

## License

[LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.html)
