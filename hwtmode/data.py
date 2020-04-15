import xarray as xr
import numpy as np
from os.path import exists, join
from glob import glob
import pandas as pd


def load_patch_files(start_date, end_date, patch_dir, input_variables, output_variables, meta_variables,
                     patch_radius=None):
    """
    Iterate through all patch files and load the input and output variables into separate py:class:`xarray.Dataset`
    objects.

    Args:
        start_date (str): Beginning of date range for file loading.
        end_date (str): End of data range for file loading.
        patch_dir (str): Path to directory containing patch netCDF files.
        input_variables (list): List of input variable names.
        output_variables (list): List of output variable names.
        meta_variables (list): List of metadata variables.
        patch_radius (int): Number of grid cells on either side of center to include.

    Returns:
        input_data (:class:`xarray.Dataset`): Datasetcontaining input variables as separate
    """
    if not exists(patch_dir):
        raise FileNotFoundError("Patch directory not found")
    patch_files = pd.Series(sorted(glob(join(patch_dir, "*.nc"))))
    date_strings = patch_files.str.split("/").str[-1].str.split("_").str[1]
    patch_dates = pd.to_datetime(date_strings)
    start_date_stamp = pd.Timestamp(start_date)
    end_date_stamp = pd.Timestamp(end_date)
    valid_patch_files = patch_files[(patch_dates >= start_date_stamp) & (patch_dates <= end_date_stamp)]
    if len(valid_patch_files) == 0:
        raise FileNotFoundError("No patch files found in " + patch_dir)
    input_data_list = []
    output_data_list = []
    meta_data_list = []
    for p, patch_file in enumerate(valid_patch_files):
        if p % 10 == 0:
            print(p, patch_file)
        ds = xr.open_dataset(patch_file)
        if patch_radius is not None:
            row_mid = int(np.round(((ds["row"].max() - ds["row"].min()) / 2).values[()]))
            col_mid = int(np.round(((ds["col"].max() - ds["col"].min()) / 2).values[()]))
            row_slice = slice(row_mid - patch_radius, row_mid + patch_radius)
            col_slice = slice(col_mid - patch_radius, col_mid + patch_radius)
            input_data_list.append(ds[input_variables].sel(row=row_slice, col=col_slice).compute())
            output_data_list.append(ds[output_variables].sel(row=row_slice, col=col_slice).compute())
            meta_data_list.append(ds[meta_variables].sel(row=row_slice, col=col_slice).compute())
        else:
            input_data_list.append(ds[input_variables].compute())
            output_data_list.append(ds[output_variables].compute())
            meta_data_list.append(ds[meta_variables].compute())

        ds.close()
    input_data = xr.concat(input_data_list, dim="p")
    output_data = xr.concat(output_data_list, dim="p")
    meta_data = xr.concat(meta_data_list, dim="p")
    return input_data, output_data, meta_data


def combine_patch_data(patch_data, variables):
    """
    Combines separate DataArrays from a Dataset into one combined DataArray for input to deep learning.

    Args:
        patch_data: :class:`xarray.Dataset` being combined.
        variables: List of variable names.

    Returns:
         combined: xarray.DataArray with dimensions (p, row, col, var_name)
    """
    combined = xr.concat([patch_data[variable] for variable in variables],
                         pd.Index(variables, name="var_name"))
    return combined.transpose("p", "row", "col", "var_name")


def min_max_scale(patch_data, scale_values=None):
    """
    Rescale the each variable in the combined DataArray to range from 0 to 1.

    Args:
        patch_data:
        scale_values:

    Returns:

    """
    fit = False
    if scale_values is None:
        scale_values = pd.DataFrame(0, index=patch_data["var_name"].values, columns=["min", "max"])
        fit = True
    transformed = patch_data.copy(deep=True)
    for v, var_name in enumerate(patch_data["var_name"].values):
        print(var_name)
        if fit:
            scale_values.loc[var_name, "min"] = patch_data[..., v].min().values[()]
            scale_values.loc[var_name, "max"] = patch_data[..., v].max().values[()]
        transformed[..., v] = (patch_data[..., v] - scale_values.loc[var_name, "min"]) \
            / (scale_values.loc[var_name, "max"] - scale_values.loc[var_name, "min"])
    return transformed, scale_values


def min_max_inverse_scale(transformed_data, scale_values):
    """
    Inverse scale data that ranges from 0 to 1 to the original values.

    Args:
        transformed_data: xarray.DataArray with data scaled from 0 to 1.
        scale_values: pandas.DataFrame containing the min and max values for each variable.

    Returns:
        xarray.DataArray with original values.
    """
    inverse = transformed_data.copy(deep=True)
    for v, var_name in enumerate(transformed_data["var_name"].values):
        inverse[..., v] = transformed_data[..., v] * (
                scale_values.loc[var_name, "max"]
                - scale_values.loc[var_name, "min"]) + scale_values.loc[var_name, "min"]
    return inverse


def storm_max_value(output_data, masks):
    return (output_data * masks).max(axis=-1).max(axis=-1).values
