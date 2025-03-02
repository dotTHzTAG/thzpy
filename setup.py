from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Data proccessing for THz-TDS'
LONG_DESCRIPTION = ('A scientific computing package for the processing of terahertz time-domain spectroscopy data. Includes modules for time-domain, transfer functions, frequency-domain, and data formatting.')

# Setting up
setup(
        name="thzpy",
        version=VERSION,
        author="Jasper Ward-Berry",
        author_email="<jnw35@cam.ac.uk>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'h5py'],
        keywords=['python',
                  'thz',
                  'terahertz',
                  'time-domain',
                  'time domain',
                  'spectroscopy'],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Research, Industry",
            "Programming Language :: Python :: 3",
            "Operating System :: Microsoft :: Windows",
        ]
)
