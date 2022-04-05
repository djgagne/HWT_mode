#!/usr/bin/env python
import numpy as np
import pandas as pd
from hagelslag.processing.ObjectMatcher import shifted_centroid_distance
from hagelslag.processing.ObjectMatcher import centroid_distance, time_distance

# NOTE: HRRR variables must be listed in the following format:
# {HRRR_VARIABLE_NAME}-{HRRR_level}
# For example, Composite Reflectivity (REFC) which is at the (entire_atmosphere) level
# would be listed as 'REFC-entire_atmosphere'

# 'ensemble_name' must be lsited as 'HRRR-ZARR'
# 'model_path' must be "hrrrzarr/sfc/"
# 'end_hour' must be no more than n-1 number of forecast hours in model

# Support for model runs with different forecast lengths, use 'HRRR_alt_end_hour' for an alternative
# forecast length that is used for each model run hour listed in 'HRRR_alt_hours'. If neither is provided,
# 'end_hour' will be used for all model runs.

## output path
scratch_path = "/glade/scratch/cbecker/HWT_2022_rt_test/"

# Historical runs
# date_index = pd.date_range(start='2021-06-14-1800', end='2021-06-14-1900', freq='1H', closed='left', tz='UTC').to_pydatetime()

# Real Time runs
# Use pd.Timedelta to correspond with delay in data availability from hour that script is submitted
date_index = pd.DatetimeIndex([pd.Timestamp.utcnow().strftime("%Y-%m-%d-%H")]) - pd.Timedelta(hours=3)

ensemble_members = ['oper']

config = dict(dates=date_index,
              start_hour=1,
              end_hour=17,
              HRRR_alt_end_hour=35,
              HRRR_alt_run_hours=[0, 6, 12, 18],
              watershed_variable="REFC-entire_atmosphere",
              ensemble_name="HRRR-ZARR",
              ensemble_members=ensemble_members,
              model_path="hrrrzarr/sfc/",
              segmentation_approach="hyst",
              model_watershed_params=(35, 50),
              size_filter=12,
              gaussian_window=1,
              mrms_path=None,
              mrms_variable="MESH_Max_60min_00.50",
              mrms_watershed_params=(13, 1, 125, 100, 100),
              object_matcher_params=([shifted_centroid_distance], np.array([1.0]),
                                     np.array([24000])),
              track_matcher_params=([centroid_distance, time_distance],
                                     np.array([80000, 2])),
              storm_variables=["REFC-entire_atmosphere", "MXUPHL_1hr_max_fcst-5000_2000m_above_ground",
                               "MXUPHL_1hr_max_fcst-3000_0m_above_ground", "MXUPHL_1hr_max_fcst-2000_0m_above_ground",
                               "HLCY-1000_0m_above_ground", "HLCY-3000_0m_above_ground", "MAXUVV_1hr_max_fcst-100_1000mb_above_ground",
                               "RELV_1hr_max_fcst-1000_0m_above_ground", "RELV_1hr_max_fcst-2000_0m_above_ground",
                               "MAXDVV_1hr_max_fcst-100_1000mb_above_ground", "MNUPHL_1hr_min_fcst-5000_2000m_above_ground",
                               "PRES-surface", "TMP-2m_above_ground", "SPFH-2m_above_ground", "DPT-2m_above_ground",
                               "UGRD-10m_above_ground", "VGRD-10m_above_ground",
                               "TCOLG_1hr_max_fcst-entire_atmosphere_single_layer",
                               "USTM-0_6000m_above_ground", "VSTM-0_6000m_above_ground",
                               "HGT-level_of_adiabatic_condensation_from_sfc"],
              potential_variables=["VUCSH-0_1000m_above_ground", "VVCSH-0_1000m_above_ground",
                                   "VUCSH-0_6000m_above_ground", "VVCSH-0_6000m_above_ground",
                                   "CAPE-180_0mb_above_ground", "CIN-180_0mb_above_ground",
                                   "CAPE-255_0mb_above_ground", "CIN-255_0mb_above_ground",
                                   "CAPE-90_0mb_above_ground", "CIN-90_0mb_above_ground",
                                   "CAPE-0_3000m_above_ground",
                                   "PRES-surface", "TMP-2m_above_ground",
                                   "SPFH-2m_above_ground", "DPT-2m_above_ground",
                                   "UGRD-10m_above_ground", "VGRD-10m_above_ground"],
              tendency_variables=[],
              shape_variables=["area", "eccentricity", "major_axis_length", "minor_axis_length", "orientation"],
              variable_statistics=["mean", "max", "min"],
              csv_path=scratch_path + "track_data_hrrr_3km_csv/",
              geojson_path=scratch_path + "track_data_hrrr_3km_json/",
              nc_path=scratch_path + "track_data_hrrr_3km_nc/",
              patch_radius=48,
              unique_matches=True,
              closest_matches=True,
              match_steps=True,
              train=False,
              single_step=True,
              label_type="gamma",
              model_map_file="/glade/u/home/dgagne/hagelslag/mapfiles/hrrr_map_2016.txt",
              mask_file=None)