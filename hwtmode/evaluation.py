import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
import scipy
from os.path import join
import xarray as xr
import joblib
from hwtmode.data import transform_data, get_gmm_predictions, load_gridded_data
from hwtmode.models import load_conv_net
from pyproj import Proj


def brier_skill_score(y_true, y_pred):
    bs_climo = np.mean((y_true.mean() - y_true) ** 2)
    bs = brier_score_loss(y_true, y_pred)
    return 1 - bs / bs_climo


def classifier_metrics(y_true, model_predictions):
    """

    Args:
        self:
        y_true:
        model_predictions:
        out_path:

    Returns:

    """
    metric_names = ["AUC", "Brier_Score", "Brier_Skill_Score"]
    metric_funcs = {"AUC": roc_auc_score,
                    "Brier_Score": brier_score_loss,
                    "Brier_Skill_Score": brier_skill_score
                    }
    metrics = pd.DataFrame(0, index=model_predictions.columns, columns=metric_names)
    for metric in metric_names:
        for model_name in model_predictions.columns:
            metrics.loc[model_name, metric] = metric_funcs[metric](y_true,
                                                                   model_predictions[model_name].values)
    return metrics


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
    idx = scipy.spatial.distance.cdist(lonlat, ll).argmin(0)  # Calculate all distances and get index of minimum

    return np.column_stack((np.unravel_index(idx, lon_array.shape))).tolist()


def combine_storm_reports(start_date, end_date, out_dir, report_type):
    """
    Combine SPC storm reports into single Pandas DataFrame.
    Args:
        start_date (str): Starting Date (format: YYYYMMDD)
        end_date (str): Ending Date (format: YYYYMMDD)
        out_dir (str): Path to download to
        report_type (str): SPC extention for specific report type (Ex. 'filtered_torn', 'hail', filtered_wind')
    """

    dates = pd.date_range(start_date, end_date).strftime('%y%m%d')
    file_list = []
    for date in dates:
        filename = join(out_dir, f'{date}_{report_type}.csv')
        f = pd.read_csv(filename)
        f['Report_Date'] = pd.to_datetime('20' + date)
        f['Actual_Date'] = f['Report_Date']
        f.loc[f['Time'] < 1200, 'Actual_Date'] += pd.Timedelta('1D')
        file_list.append(f)
    df = pd.concat(file_list)
    hours = df.loc[:, 'Time'].apply(lambda x: str(x)[:-2] if len(str(x)) >= 3 else '0').astype(int) + 1
    df['Actual_Date'] = df['Actual_Date'] + pd.to_timedelta(hours, unit='h')
    return df


def generate_obs_grid(beg, end, obs_path, model_grid_path):
    """
    Generate Xarray dataset (Time, x, y) of observed storm reports. Each hazard is stored as a seperate variable. Valid time is separted by hour. Minimum and maximum
    lead times are used for ensembled HRRR runs.
    Args:
        beg (str): Beginning of date range (format: YYMMDDhhmm)
        end (str): End of date range (format: YYMMDDhhmm)
        obs_path: Path to downloaded storm reports
        model_grid_path: Path to coarse grid
    Returns:
        Xarray dataset (Time, x, y) of storm report counts. Different hazards represented as variables.
    """
    grid = xr.open_dataset(model_grid_path)
    valid_dates = pd.date_range(pd.Timestamp(beg), pd.Timestamp(end), freq='1h')
    obs_list = []

    for report_type in ['filtered_torn', 'filtered_wind', 'filtered_hail']:

        ds_list = []

        obs = combine_storm_reports(beg, end, obs_path, report_type)

        for valid_date in valid_dates:

            ds = grid.expand_dims('time').assign_coords(valid_time=('time', [valid_date]))
            ds[report_type.split('_')[-1]] = ds['lat'] * 0

            obs_sub = obs[obs['Actual_Date'] == valid_date]
            obs_indx = find_coord_indices(ds['lon'].values,
                                          ds['lat'].values,
                                          obs_sub['Lon'],
                                          obs_sub['Lat'])
            for i in obs_indx:
                if i is not None:
                    ds[report_type.split('_')[-1]][i[0], i[1]] += 1
                else:
                    continue
            ds_list.append(ds)

        obs_list.append(xr.concat(ds_list, dim='time'))

    return xr.merge(obs_list)


def generate_storm_grid(beg, end, label_path, model_list, model_grid, min_lead_time, max_lead_time, file_format):
    """
    Populates empty grid in any grid cell that contained a storm in the given date range.
    Args:
        beg (str): beginning date string for neighborhood probability (format: 'YYYYMMDDHHHH')
        end (str): ending date string for neighborhood probability (format: 'YYYYMMDDHHHH') (can be same or after beg)
        label_path (str): path to pickle files
        model_list (list): List of model names
        model_grid (netCDF): Model grid
        min_lead_time (int): Minimum model lead time
        max_lead_time (int): Maximum model lead time
        file_format (str): File format of saved labels

    Returns:
        populated storm grid
    """
    storm_grid = model_grid.copy()
    dates = pd.date_range(pd.Timestamp(beg) - pd.Timedelta(hours=max_lead_time),
                          pd.Timestamp(end) - pd.Timedelta(hours=min_lead_time), freq='1h')
    for model in model_list:
        df_storms, storm_indxs = {}, {}
        df_list = []
        for date in dates:
            file_string = join(label_path, f"{model}_labels_{date.strftime('%Y%m%d-%H00')}.{file_format}")
            try:
                if file_format == 'pkl':
                    df = pd.read_pickle(file_string)
                elif file_format == 'csv':
                    df = pd.read_csv(file_string)
                    df['time'] = df['time'].astype('datetime64')
                    df['run_date'] = df['run_date'].astype('datetime64')
                elif file_format == 'parquet':
                    df = pd.read_parquet(file_string)
            except:
                print(f"{file_string} doesn't seem to exist. Skipping.")
                continue

            df_list.append(df[(df['time'] >= pd.Timestamp(beg)) & (df['time'] <= pd.Timestamp(end))])

        d = pd.concat(df_list)
        for storm_type in ['Disorganized', 'QLCS', 'Supercell']:

            if storm_type == 'all':
                df_storms[storm_type] = d
            else:
                df_storms[storm_type] = d[d['label'] == storm_type]

            storm_indxs[storm_type] = find_coord_indices(model_grid['lon'].values,
                                                         model_grid['lat'].values,
                                                         df_storms[storm_type]['centroid_lon'],
                                                         df_storms[storm_type]['centroid_lat'])
            storm_grid[f'{model}_{storm_type}'] = storm_grid['lat'] * 0
            for i in storm_indxs[storm_type]:
                storm_grid[f'{model}_{storm_type}'][i[0], i[1]] = 1
    return


def gridded_probabilities(run_date, forecast_hour, data_dir, ML_model_path, ML_model_name, physical_model, input_vars,
                          output_vars, meta_vars, patch_radius, stride, gmm=False):
    """
    Generate model probabilities across entire grid (compatabile with WRF and HRRR).
    Args:
        run_date: model run date in format YYYYMMDD (or "today")
        forecast_hour: Model forecast Hour in format HH
        data_dir: Base directory for Raw WRF output or AWS S3 bucket location for HRRR data.
        ML_model_path: Base path of model.
        ML_model_name: Name of model.
        physical_model: Model used to generate data. Supports 'HRRR' or 'WRF'.
        input_vars: Input variables. Must match those for model data was generated with. For HRRR, use format {variable_name}-{name_of_level}
        output_vars: Output variables.
        meta_vars: Meta variables to be kept.
        patch_radius: Radius of patches CNN model was trained on.
        stride: Window stride in both x and y directions to sample patches from full grid
        gmm (boolean): Use Gaussian Mixture Model. If False, default to CNN.
    Returns:
        (List) Containing each model run.
    """

    mod = load_conv_net(join(ML_model_path, ML_model_name), ML_model_name)
    scale_values = pd.read_csv(join(ML_model_path, ML_model_name, f"scale_values_{ML_model_name}.csv"))
    scale_values['variable'] = [i.split('-')[0] for i in input_vars]
    scale_values = scale_values.set_index('variable')
    if gmm:
        gmm_mod = joblib.load(join(ML_model_path, ML_model_name, f'{ML_model_name}.gmm'))
        cluster_assignments = joblib.load(join(ML_model_path, ML_model_name, f'{ML_model_name}_gmm_labels.dict'))
    patch_size = patch_radius * 2 + 1
    mode_types = ['QLCS', 'Supercell', 'Disorganized']
    daily_list = []
    ds = load_gridded_data(data_dir, physical_model, run_date, forecast_hour, input_vars, output_vars, meta_vars)
    ds_coarse = ds.isel(row=slice(patch_size + 1, ds.row.size, stride), col=slice(patch_size + 1, ds.col.size, stride))
    input_vars = [x.split("-")[0] for x in input_vars]
    for mode_type in mode_types:
        ds_coarse[f'{mode_type}'] = ds_coarse[input_vars[0]] * np.nan
        ds_coarse[f'{mode_type}_prob'] = ds_coarse[input_vars[0]] * np.nan
    if physical_model.upper() == 'HRRR':
        reflectivity = 'REFC'
    elif physical_model.upper() == 'WRF':
        reflectivity = 'REFL_COM'
    for time_i in range(len(ds_coarse['Time'])):
        print(time_i)
        ns, we, = ds['row'].size, ds['col'].size
        for i_count, i in enumerate(np.arange(0, ns - patch_size, stride)):
            for j_count, j in enumerate(np.arange(0, we - patch_size, stride)):
                y, x = slice(i, i + patch_size), slice(j, j + patch_size)
                p = ds.isel(Time=slice(time_i, time_i + 1), row=y, col=x)
                if p[reflectivity].values.max() > 35:
                    p_transformed = transform_data(p, input_vars, scale_values)
                    if gmm:
                        preds = get_gmm_predictions(p_transformed, mod, gmm_mod, cluster_assignments)
                    else:
                        preds = mod.predict(p_transformed)
                    ds_coarse['QLCS_prob'].loc[dict(Time=time_i, row=i_count, col=j_count)] = preds[0][0]
                    ds_coarse['Supercell_prob'].loc[dict(Time=time_i, row=i_count, col=j_count)] = preds[0][1]
                    ds_coarse['Disorganized_prob'].loc[dict(Time=time_i, row=i_count, col=j_count)] = preds[0][2]
    ds_coarse['QLCS'].values = np.where((ds_coarse['QLCS_prob'] > ds_coarse['Supercell_prob']) &
                                        (ds_coarse['QLCS_prob'] > ds_coarse['Disorganized_prob']), 1, np.nan)
    ds_coarse['Supercell'].values = np.where((ds_coarse['Supercell_prob'] > ds_coarse['QLCS_prob']) &
                                             (ds_coarse['Supercell_prob'] > ds_coarse['Disorganized_prob']), 1, np.nan)
    ds_coarse['Disorganized'].values = np.where((ds_coarse['Disorganized_prob'] > ds_coarse['Supercell_prob']) &
                                                (ds_coarse['Disorganized_prob'] > ds_coarse['QLCS_prob']), 1, np.nan)
    daily_list.append(ds_coarse)
    del ds_coarse

    return daily_list


