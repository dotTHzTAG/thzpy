from dataclasses import dataclass, field
from typing import Dict, Any
from collections.abc import Iterable
from warnings import warn
import numpy as np
import h5py


@dataclass
class DotthzMetaData:
    """Data class holding metadata for measurements in the .thz file format.

    Attributes
    ----------
    user : str
        The user who performed the measurement.
    email : str
        The email of the user.
    orcid : str
        The ORC ID of the user.
    institution : str
        The institution the user belongs to.
    description : str
        A description of the measurement.
    md : dict
        A dictionary of custom metadata (e.g. thickness, temperature, etc.).
    version : str, optional
        The version of the .thz file standard used.
        Defaults to "1.00".
    mode : str
        The measurement modality (e.g. transmission).
    instrument : str
        The instrument used to perform the measurement.
    time : str
        Timestamp of when the measurement was conducted.
    date : str
        The date on which the measurement was conducted.
    """
    user: str = ""
    email: str = ""
    orcid: str = ""
    institution: str = ""
    description: str = ""
    md: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.00"
    mode: str = ""
    instrument: str = ""
    time: str = ""
    date: str = ""

    def add_field(self, key, value):
        self.md[key] = value


@dataclass
class DotthzMeasurement:
    """Data class for terahertz time-domain spectroscopy measurements.

    Holds a dictionary of datasets and a metadata object.

    Attributes
    ----------
    datasets : dict of array like
        A dictionary of datasets from a measurement (e.g. waveforms).
    metadata : DotthzMetaData
        Object containing the measurement metadata.
    """
    datasets: Dict[str, np.ndarray] = field(default_factory=dict)
    meta_data: DotthzMetaData = field(default_factory=DotthzMetaData)


class DotthzFile:
    """
    File class for the .thz format, holding THz time-domain spectroscopy data.

    Holds a dictionary of measurment objects and a link to a base file. If the
    object is initilised in the read or append modes with a path to a valid
    .thz file then any measurments in that file will be loaded to the object.

    Attributes
    ----------
    measurements : dict of DotthzMeasurement
        Dictionary containing thz measurement objects.
    file : h5py.File
        The base file where measurements are read/written.

    Methods
    -------
    load(path)
        Load measurements from a .thz file at the path to the file object.
    get_measurements
        Return a dict of all measurements in the file object.
    get_measurement_names
        Return a list of all measurement names in the file object.
    get_measurement(name)
        Return the specified measurement from the file object.
    write_measurement(name, measurement)
        Write a measurement to the file object.

    """
    def __init__(self, name, mode="r", driver=None, libver=None,
                 userblock_size=None, swmr=False, rdcc_nslots=None,
                 rdcc_nbytes=None, rdcc_w0=None, track_order=None,
                 fs_strategy=None, fs_persist=False, fs_threshold=1,
                 fs_page_size=None, page_buf_size=None, min_meta_keep=0,
                 min_raw_keep=0, locking=None, alignment_threshold=1,
                 alignment_interval=1, meta_block_size=None, **kwds):
        self.measurements = {}
        self.file = h5py.File(name, mode, driver=driver, libver=libver,
                              userblock_size=userblock_size, swmr=swmr,
                              rdcc_nslots=rdcc_nslots, rdcc_nbytes=rdcc_nbytes,
                              rdcc_w0=rdcc_w0, track_order=track_order,
                              fs_strategy=fs_strategy, fs_persist=fs_persist,
                              fs_threshold=fs_threshold,
                              fs_page_size=fs_page_size,
                              page_buf_size=page_buf_size,
                              min_meta_keep=min_meta_keep,
                              min_raw_keep=min_raw_keep, locking=locking,
                              alignment_threshold=alignment_threshold,
                              alignment_interval=alignment_interval,
                              meta_block_size=meta_block_size, **kwds)

        if "r" in mode or "a" in mode:
            self._load(self.file)

    def __enter__(self):
        # Enable the use of the `with` statement.

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Close any resources if applicable.

        if self.file is not None:
            self.file.close()
            self.file = None

    def _get_descriptions(self, desc_in):
        # Handles inconstistent formatting for metadata descriptions.

        desc_in = self._sanatize(desc_in)

        if isinstance(desc_in, str):
            desc_list = [desc.strip() for desc in desc_in.split(",")]
        else:
            if not isinstance(desc_in, Iterable):
                desc_in = [desc_in]

            try:
                desc_list = list(map(str, desc_in))
            except (TypeError, ValueError):
                desc_list = []
                warn("Could not import descriptions.")

        return desc_list

    def _sanatize(self, md_in):
        # Reduces reduntant iterables to base data.

        if (isinstance(md_in, Iterable) and len(md_in) == 1):
            return self._sanatize(md_in[0])
        else:
            return md_in

    def _load(self, file):
        # Load data from a .thz file.

        groups = {}
        for group_name, group in file.items():
            measurement = DotthzMeasurement()

            # Load datasets.
            if "dsDescription" in group.attrs:
                ds_description_attr = group.attrs["dsDescription"]
                ds_descriptions = self._get_descriptions(ds_description_attr)

                for i, desc in enumerate(ds_descriptions):
                    dataset_name = f"ds{i + 1}"
                    if dataset_name in group:
                        measurement.datasets[desc] = group[dataset_name][...]

            # Load metadata attributes.
            for attr in ["description", "date", "instrument",
                         "mode", "time", "thzVer"]:
                if attr in group.attrs:
                    setattr(measurement.meta_data,
                            attr,
                            self._sanatize(group.attrs[attr]))

            # Special handling for user metadata.
            if "user" in group.attrs:
                user_info = self._sanatize(group.attrs["user"]).split("/")
                fields = ["orcid", "user", "email", "institution"]
                for i, part in enumerate(user_info):
                    if i < len(fields):
                        setattr(measurement.meta_data, fields[i], part)

            # Lead measurement metadata.
            if "mdDescription" in group.attrs:
                md_description_attr = group.attrs["mdDescription"]
                md_descriptions = self._get_descriptions(md_description_attr)

                for i, desc in enumerate(md_descriptions):
                    md_name = f"md{i + 1}"
                    if md_name in group.attrs:
                        md_val = self._sanatize(group.attrs[md_name])
                        try:
                            measurement.meta_data.md[desc] = float(md_val)
                        except (ValueError, TypeError):
                            measurement.meta_data.md[desc] = md_val

            groups[group_name] = measurement

        self.measurements.update(groups)

    def load(self, path):
        """Load measurements from a .thz file at the path to the file object.

        Parameters
        ----------
        path : str
            The path to the file.
        """
        file = h5py.File(path, 'r')
        self._load(file)
        file.close()

    def get_measurements(self):
        "Return a dict of all measurements in the file object."
        return self.measurements

    def get_measurement_names(self):
        """Return a list of all measurement names in the file object."""
        return list(self.measurements.keys())

    def get_measurement(self, name):
        """Return the specified measurement from the file object.

        Parameters
        ----------
        name : str
            The name of the measurement.

        Returns
        -------
        DotthzMeasurement
            The requested measurement
        """
        return self.measurements.get(name)

    def write_measurement(self, name: str,
                          measurement: DotthzMeasurement):
        """Write a measurement to the file object.

        Parameters
        ----------
        name : str
            The name of the measurement.
        measurement : DotthzMeasurement
            The measurement to be added to the file object.
        """
        group = self.file.create_group(name)

        # Write dataset descriptions
        ds_descriptions = ", ".join(measurement.datasets.keys())
        group.attrs["dsDescription"] = ds_descriptions

        # Write datasets
        for i, (name, dataset) in enumerate(measurement.datasets.items()):
            ds_name = f"ds{i + 1}"
            group.create_dataset(ds_name, data=dataset)

        # Write metadata
        for attr_name, attr_value in measurement.meta_data.__dict__.items():
            if attr_name == "md":
                # Write md descriptions as an attribute
                md_descriptions = ", ".join(measurement.meta_data.md.keys())
                group.attrs["mdDescription"] = md_descriptions
                for i, md_val in enumerate(measurement.meta_data.md.values()):
                    md_name = f"md{i + 1}"
                    try:
                        # Attempt to save as float if possible
                        group.attrs[md_name] = float(md_val)
                    except (ValueError, TypeError):
                        group.attrs[md_name] = md_val
            elif attr_name == "version":
                group.attrs["thzVer"] = measurement.meta_data.version

            elif attr_name in ["orcid", "user", "email", "institution"]:
                continue
            else:
                if attr_value:  # Only write non-empty attributes
                    group.attrs[attr_name] = attr_value

        # Write user metadata in the format "ORCID/user/email/institution"
        user_info = "/".join([
            measurement.meta_data.orcid,
            measurement.meta_data.user,
            measurement.meta_data.email,
            measurement.meta_data.institution
        ])
        group.attrs["user"] = user_info
