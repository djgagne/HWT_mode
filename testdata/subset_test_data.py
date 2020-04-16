import xarray as xr
from os.path import join, exists
from os import makedirs
from glob import glob

input_directory = "track_data_ncarstorm_3km_REFL_COM_ws_nc"
output_directory = "track_data_ncarstorm_3km_REFL_COM_ws_nc_small"
patch_files = sorted(glob(join(input_directory, "*.nc")))
input_variables = ["REFL_1KM_AGL_curr", "U10_curr", "V10_curr"]
output_variable = ["UP_HELI_MAX_curr"]
meta_variables = ["masks", "i", "j", "time", "centroid_lon", "centroid_lat",
                       "centroid_i", "centroid_j", "track_id", "track_step"]
if not exists(output_directory):
    makedirs(output_directory)
for patch_file in patch_files:
    print(patch_file)
    ds = xr.open_dataset(patch_file)
    ds_sub = ds[input_variables + output_variable + meta_variables]
    encoding = {}
    for var in input_variables + output_variable + meta_variables:
        encoding[var] = {'zlib': True, 'complevel': 5, "shuffle": True, "least_significant_digit": 3}
    ds_sub.to_netcdf(join(output_directory, patch_file.split("/")[-1]), encoding=encoding)
