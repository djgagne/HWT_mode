import pandas as pd
import numpy as np
import xarray as xr
from os.path import join, isfile
import urllib
from scipy.ndimage.filters import gaussian_filter
import s3fs
import joblib
from pyproj import Proj
from scipy.spatial.distance import cdist

def find_coord_indices(lon_array, lat_array, lon_points, lat_points, dist_proj='lcc'):
    """
    Find indices of nearest lon/lat pair on a grid. Supports rectilinear and curilinear grids.
    lon_points / lat_points must be received as a list.
    Args:
        lon_array (np.array): Longitude values of coarse grid you are matching against
        lat_array (np.array): Latitude values of coarse grid you are matching against
        lon_points (list): List of Longitude points from orginal grid/object
        lat_points (list): List of Latitude points from original grid/object
        dist_proj (str): Name of projection for pyproj to calculate distances
    Returns (list):
        List of i, j (Lon/Lat) indices for coarse grid.

    """
    if dist_proj == 'lcc':
        proj = Proj(proj='lcc', R=6371229, lat_0=38.336433, lon_0=-97.53348, lat_1=32, lat_2=46)  ## from WRF HWT data

    proj_lon, proj_lat = np.array(proj(lon_array, lat_array))  # transform to distances using specified projection
    lonlat = np.column_stack(
        (proj_lon.ravel(), proj_lat.ravel()))  # Stack all coarse x, y distances for array shape (n, 2)
    ll = np.array(proj(lon_points, lat_points)).T  # transform lists of fine grid x, y to match shape (n, 2)
    idx = cdist(lonlat, ll).argmin(0)  # Calculate all distances and get index of minimum

    return np.column_stack((np.unravel_index(idx, lon_array.shape))).tolist()


def fetch_storm_reports(start_date, end_date, out_dir, report_type):
    """
    Download SPC storm reports.
    Args:
        start_date (str): Starting Date (format: YYYYMMDD)
        end_date (str): Ending Date (format: YYYYMMDD)
        out_dir (str): Path to download to
        report_type (str): SPC extention for specific report type (Ex. 'filtered_torn', 'hail', filtered_wind')
    """

    dates = pd.date_range(start_date, end_date).strftime('%y%m%d')
    for date in dates:
        if isfile(join(out_dir, f'{date}_{report_type}.csv')):
            continue
        else:
            url = f'https://www.spc.noaa.gov/climo/reports/{date}_rpts_{report_type}.csv'
            filename = join(out_dir, f'{date}_{report_type}.csv')
        try:
            urllib.request.urlretrieve(url, filename)
        except:
            pass

def combine_storm_reports(start_date, end_date, out_dir, report_type):
    """
    Combine SPC storm reports into single Pandas DataFrame.
    Args:
        start_date (str): Starting Date (format: YYYYMMDD)
        end_date (str): Ending Date (format: YYYYMMDD)
        out_dir (str): Path to download to
        report_type (str): SPC extention for specific report type (Ex. 'filtered_torn', 'hail', filtered_wind')
    """

    dates = pd.date_range(start_date, end_date)
    file_list = []
    for date in dates:
        if not isfile(join(out_dir, f"{date.strftime('%y%m%d')}_{report_type}.csv")):
            continue
        filename = join(out_dir, f"{date.strftime('%y%m%d')}_{report_type}.csv")
        f = pd.read_csv(filename)
        f['Report_Date'] = pd.to_datetime(date.strftime('%Y%m%d'))
        f['Actual_Date'] = f['Report_Date']
        f.loc[f['Time'] < 1200, 'Actual_Date'] += pd.Timedelta('1D')
        file_list.append(f)
    df = pd.concat(file_list)
    hours = df.loc[:, 'Time'].apply(lambda x: str(x)[:-2] if len(str(x)) >= 3 else '0').astype(int) + 1
    df['Actual_Date'] = df['Actual_Date'] + pd.to_timedelta(hours, unit='h')
    return df


def generate_obs_grid(beg, end, storm_report_path, model_grid_path):
    """
    Generate Xarray dataset (Time, x, y) of observed storm reports. Each hazard is stored as a seperate variable. Valid time is separted by hour. Minimum and maximum
    lead times are used for ensembled HRRR runs.
    Args:
        beg (str): Beginning of date range (format: YYMMDDhhmm)
        end (str): End of date range (format: YYMMDDhhmm)
        storm_report_path: Path to downloaded storm reports
        model_grid_path: Path to coarse grid
    Returns:
        Xarray dataset (Time, x, y) of storm report counts. Different hazards represented as variables.
    """
    grid = xr.open_dataset(model_grid_path)
    valid_dates = pd.date_range(pd.Timestamp(beg), pd.Timestamp(end), freq='1h', closed='right')
    obs_list = []

    for report_type in ['filtered_torn', 'filtered_wind', 'filtered_hail']:

        ds_list = []

        obs = combine_storm_reports(beg, end, storm_report_path, report_type)

        for valid_date in valid_dates:

            ds = grid.expand_dims('time').rename({'time': 'Time'}).assign_coords(valid_time=('Time', [valid_date]))
            ds[report_type.split('_')[-1]] = ds['lat'] * 0

            obs_sub = obs[obs['Actual_Date'] == valid_date]
            obs_indx = find_coord_indices(ds['lon'].values, ds['lat'].values, obs_sub['Lon'], obs_sub['Lat'])
            for i in obs_indx:
                if i is not None:
                    ds[report_type.split('_')[-1]][i[0], i[1]] += 1
                else:
                    continue
            ds_list.append(ds)

        obs_list.append(xr.concat(ds_list, dim='Time'))

    return xr.merge(obs_list)


def generate_mode_grid(beg, end, labels, model_grid_path, min_lead_time, max_lead_time, run_date_freq='1h',
                       bin_width=None):
    """
    Convert tabular ML storm mode predictions and probabilites to a coarse gridded product (Xarray dataset) with dimentions (Time, y, x) and associated
    neighborhood probabilities. Supports a storm surrogate probability function (SSPF) by using min/max lead times.
    Args:
        beg: Beginning of date range (format: 'YYMMDDhhmm')
        end: End of date range (format: 'YYMMDDhhmm')
        label_path: Path to labels/ predictions
        model: Model name (encoded into label files)
        model_grid_path: Path to coarse grid
        min_lead_time: Minimum leadtime for overlapping ensembles produced by HRRR
        max_lead_time: Maximum lead time for over lapping ensembled produced by HRRR
        run_date_freq: Frequency spacing of model run times ('{X}h', '{x}d', ...)
        bin_width: Width of bins for ML probabilities
    Returns:
        Xarray dataset (Time, x, y) of storm object counts. Different modes represented as variables.
    """

    storm_grid = xr.open_dataset(model_grid_path)
    valid_dates = pd.date_range(pd.Timestamp(beg) + pd.Timedelta(hours=1), pd.Timestamp(end), freq='1h')
    df_storms, storm_indxs = {}, {}
    df_list, ds_list = [], []

    for valid_date in valid_dates:
        d_sub = labels[labels['time'] == valid_date]
        ds = storm_grid.expand_dims('time').assign_coords(valid_time=('time', [valid_date]))

        for storm_type in ['Supercell', 'QLCS', 'Disorganized']:

            df_storms[storm_type] = d_sub[d_sub['label'] == storm_type]

            storm_indxs[storm_type] = find_coord_indices(ds['lon'].values,
                                                         ds['lat'].values,
                                                         df_storms[storm_type]['centroid_lon'],
                                                         df_storms[storm_type]['centroid_lat'])
            ds[storm_type] = ds['lat'] * 0
            for i in storm_indxs[storm_type]:
                ds[storm_type][i[0], i[1]] += 1

            ds = compute_neighborhood_prob(ds, storm_type)
            ds[storm_type].attrs['Description'] = f'Categorical Classification of {storm_type}'
            ds[storm_type + '_nprob'].attrs['Description'] = f'Neighborhood Probabilities for {storm_type}'

            if bin_width is not None:

                bins = np.arange(0, 1 + bin_width, bin_width)
                for indx in range(len(bins) - 1):
                    low, high = bins[indx], bins[indx + 1]
                    full_name = f'{storm_type}_{int(low * 100)}_{int(high * 100)}'
                    df_storms[full_name] = d_sub[
                        (d_sub[f'{storm_type}_prob'] > low) & (d_sub[f'{storm_type}_prob'] <= high)]

                    storm_indxs[full_name] = find_coord_indices(ds['lon'].values,
                                                                ds['lat'].values,
                                                                df_storms[full_name]['centroid_lon'],
                                                                df_storms[full_name]['centroid_lat'])
                    ds[full_name] = ds['lat'] * 0
                    for i in storm_indxs[full_name]:
                        ds[full_name][i[0], i[1]] += 1

                    ds = compute_neighborhood_prob(ds, full_name)
                    ds[full_name].attrs['ML Prob Bin'] = f'{low} - {high} for {storm_type}'
                    ds[full_name + '_nprob'].attrs[
                        'Description'] = f'Neighborhood probabilities for ML probabilities in the ({low} - {high}) range for {storm_type}'

        ds_list.append(ds)

    ds_all = xr.concat(ds_list, dim='time')

    return ds_all


def compute_neighborhood_prob(ds, var, sigma=1, use_binary=True):
    """
    Compute neighborhood probabilities using a gaussian filter on a binary grid of storm locations.
    Args:
        ds (xarray dataset): Dataset containing binary grid of storm locations
        var (str): Variable to compute
        sigma: Sigma for gaussian filter (smaller will make tighter contours)
        use_binary: Whether or not to convert grid cells with > 1 storms to 1.
    Returns: Xarray with additional variable containing neighborhood probabilitites
    """
    if use_binary:
        ds[var] = ds[var].expand_dims('time')
        ds[f'{var}_nprob'] = ds[var].where(ds[var] <= 1, 1).groupby('time').apply(gaussian_filter, sigma=sigma)
    else:
        ds[var] = ds[var].expand_dims('time')
        ds[f'{var}_nprob'] = ds[var].groupby('time').apply(gaussian_filter, sigma=sigma)

    return ds


def load_hrrr_data(bucket, run_date, run_hour, variables):
    """"
    Load entire HRRR forecast, including coordinate data, for specified run date from an AWS bucket.
    Args:
        bucket (str): AWS bucket name
        run_date (str): Run date string in format YYYYMMDDHH
        run_hour (str): Run Hour (HH)
        variables (list): List of variables to extract in format level-variable_name
    Returns:
        Xarray dataset of HRRR forecast
    """
    run_date_str = pd.to_datetime(run_date).strftime("%Y%m%d")
    forecast_hour_str = pd.to_datetime(join(run_date, run_hour)).strftime("%H")
    datasets = []

    for i, variable in enumerate(variables):
        files = []
        level = variable.split('-')[1]
        variable = variable.split('-')[0]
        fs = s3fs.S3FileSystem(anon=True)
        if i == 0:
            coord_path = join(bucket, run_date_str, f'{run_date_str}_{forecast_hour_str}z_fcst.zarr', level, variable)
            coords = s3fs.S3Map(root=coord_path, s3=fs, check=False)
        path = join(bucket, run_date_str, f'{run_date_str}_{forecast_hour_str}z_fcst.zarr', level, variable, level)
        f = s3fs.S3Map(root=path, s3=fs, check=False)
        files.append(coords)
        files.append(f)
        ds = xr.open_mfdataset(files, engine='zarr').load()
        ds[variable] = ds[variable].astype('float32')
        datasets.append(ds)

    all_ds = xr.merge(datasets).drop(['projection_x_coordinate', 'projection_y_coordinate', 'forecast_period'])
    all_ds = all_ds.rename_dims(dict(projection_x_coordinate='x', projection_y_coordinate='y', time='Time'))

    all_ds = all_ds.reset_coords('time').rename(dict(time='valid_time'))

    return all_ds


def load_HRRR_proxy_data(beg, end, freq, variables, max_forecast_len, AWS_bucket, HRRR_model_map):
    """
    Loads HRRR_data from AWS Bucket for specified variables and date ranges (entire CONUS grid).
    Converts <= 0 values to NaN.
    Args:
        beg: Start Date of data to be loaded (YYYYMMDDHH00)
        end: End Date of data to be loaded (YYYYMMDDHH00)
        variables: Variables to be loaded (var_name-level_name)
        max_forecast_len: Maximum forecast length
        AWS_bucket: Path to AWS bucket (Zarr format)
        HRRR_model_map: Path to HRRR model map with longitude / latitude
    Returns: Concatenated xarray dataset
    """

    hrrr_ll = xr.open_dataset(HRRR_model_map)
    proxy_ds_list = []

    for date in pd.date_range(beg, end, freq=freq, closed='left'):
        date_str = date.strftime("%Y%m%d")
        hour_str = date.strftime("%H")
        print(f'Loading HRRR data for model run initialized at {date_str}-{hour_str}Z')
        ds = load_hrrr_data(AWS_bucket, date_str, hour_str, variables)
        if len(ds['Time'] > max_forecast_len):
            ds = ds.sel(Time=slice(0, max_forecast_len))
        proxy_ds_list.append(ds)
    proxy_ds = xr.concat(proxy_ds_list, dim='Time')
    proxy_ds = proxy_ds.assign_coords(dict(
                                        lon=hrrr_ll['longitude'].astype('float32'),
                                        lat=hrrr_ll['latitude'].astype('float32'),
                                        valid_time=proxy_ds['valid_time'],
                                        forecast_reference_time=proxy_ds['forecast_reference_time']))

    for var in [x.split('-')[0] for x in variables]:
        proxy_ds[var].values = np.where(proxy_ds[var] <= 0, np.nan, proxy_ds[var])

    return proxy_ds


def load_WRF_proxy_data(beg, end, variables, max_forecast_len, WRF_data_path):
    """
    Loads WRF data for specific variables and specified time frame.
    Args:
        beg: Start Date of data to be loaded (YYYYMMDDHH00)
        end: End Date of data to be loaded (YYYYMMDDHH00)
        variables: Variables names to be loaded
        max_forecast_len: Maximum forecast length
        WRF_data_path: Base path to WRF data
    Returns: Concatenated xarray dataset

    """
    WRF_list = []
    for date in pd.date_range(beg, end, freq='1d', closed='left'):
        date_str = date.strftime('%Y%m%d%H')
        first_valid = date + pd.Timedelta('1h')
        last_valid = first_valid + pd.Timedelta(f'{max_forecast_len}h')
        print(f'Loading WRF data from model run initialized on {date}')
        for valid_hour in pd.date_range(first_valid.strftime('%Y%m%d%H%M'), last_valid.strftime('%Y%m%d%H%M'),
                                        freq='1h', closed='left'):
            year = valid_hour.strftime('%Y')
            month = valid_hour.strftime('%m')
            day = valid_hour.strftime('%d')
            hour = valid_hour.strftime('%H')
            data = xr.open_dataset(join(WRF_data_path, date_str, f'diags_d01_{year}-{month}-{day}_{hour}_00_00.nc'))[
                variables + ['Times']].load()
            data['XLONG'], data['XLAT'] = data['XLONG'].squeeze(), data['XLAT'].squeeze()
            WRF_list.append(data)
    merged_data = xr.concat(WRF_list, dim='Time')
    merged_data['valid_time'] = (
        'Time', pd.to_datetime(merged_data['Times'].astype(str).values, format='%Y-%m-%d_%H:00:00'))
    merged_data = merged_data.assign_coords({'valid_time': merged_data['valid_time'], 'Times': merged_data['Times']})
    merged_data = merged_data.rename({'XLAT': 'lat', 'XLONG': 'lon'})

    return merged_data


def get_quantiles(data, quantiles):
    """
    Get quantile values for entire dataset for each variable.
    Args:
        data: Xarray dataset
        quantiles: List of quantile values

    Returns:
        Pandas Dataframe of quantile values for each variable
    """
    variables = list(data.data_vars)
    proxy_quant_df = pd.DataFrame(columns=[str(i) for i in quantiles], index=variables)
    for var in variables:
        for q in quantiles:
            proxy_quant_df.loc[var, str(q)] = data[var].quantile(q).values

    return proxy_quant_df


def get_proxy_events(data, quantile_df, variables, model_grid_path, use_saved_indices=False, index_path=None):
    """
    Map proxy "events" onto coarse grid for each location that exceeds specific quantile.
    Args:
        data: Proxy data (xarray dataset)
        quantile_df: Dataframe of quantiles / variables
        variables: List of variables
        model_grid_path: Path to coarse grid to aggregate to
        use_saved_indices: Whether or not to load list of indices for each grid cell from fine to coarse grid
        index_path: Path to saved indices (default = None)

    Returns:
        Aggregated xarray dataset of proxy events (exceedence of variables above quantiles)
    """
    indices = get_indices(data, model_grid_path, use_saved_indices, index_path)

    coarse_grid = xr.open_dataset(model_grid_path)
    dummy_var = variables[0].split('-')[0]
    i_grid, j_grid = np.indices(data[dummy_var].shape[1:])
    i_grid, j_grid = i_grid.ravel(), j_grid.ravel()
    proxy_events = coarse_grid.copy()

    for var in [x.split('-')[0] for x in variables]:
        print(f'Converting proxy values from original to coarse grid for {var}')
        for q in quantile_df.columns:
            arr = np.zeros((data[dummy_var].shape[0], coarse_grid.dims['y'], coarse_grid.dims['x']))
            ds = data[var].values
            thresh = quantile_df.loc[var, str(q)]

            for n, (i, j) in enumerate(indices):
                arr[:, i, j] = np.where(ds[:, i_grid[n], j_grid[n]] > thresh, 1, 0)

            proxy_events[f'{var}_exceed_{q}'] = (['Time', 'y', 'x'], arr)

    return proxy_events.assign_coords({'valid_time': data['valid_time']})


def get_indices(data, coarse_grid_path, use_saved_indices=False, index_path=None):
    """
    Generate list of matching lat/lon indices on coarse grid from fine grid. Supports preloaded indices.
    Args:
        data: Proxy data (xarray dataset including lat/lon)
        coarse_grid_path: Path to coarse grid
        use_saved_indices: Boolean for using preloaded indices.
        index_path: Path to preloaded indices.

    Returns:
        List of indices on coarse grid.
    """
    if use_saved_indices:
        return joblib.load(index_path)
    else:
        coarse_grid = xr.open_dataset(coarse_grid_path)
        lat, lon = data['lat'].values, data['lon'].values
        coarse_lat, coarse_lon = coarse_grid['lat'].values, coarse_grid['lon'].values
        indices = find_coord_indices(coarse_lon, coarse_lat, lon.ravel(), lat.ravel())

        return indices