import xarray as xr
from os.path import exists, join
from glob import glob
import numpy as np


def load_patch_files(patch_dir, input_variables, output_variables):
    """
    Iterate through all patch files and load the input and output variables.

    Args:
        patch_dir: Path to directory containing patch netCDF files.
    """
    if not exists(patch_dir):
        raise FileNotFoundError("Patch directory not found")
    patch_files = sorted(glob(join(patch_dir, "*.nc")))
    if len(patch_files) == 0:
        raise FileNotFoundError("No patch files found in " + patch_dir)
    input_data_list = []
    output_data_list = []
    for p, patch_file in enumerate(patch_files):
        if p % 10 == 0:
            print(p, patch_file)
        ds = xr.open_dataset(patch_file)
        input_data_list.append(ds[input_variables].compute())
        output_data_list.append(ds[output_variables].compute())
        ds.close()
    input_data = xr.concat(input_data_list, dim="p")
    output_data = xr.concat(output_data_list, dim="p")
    return input_data, output_data
