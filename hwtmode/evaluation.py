import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
import scipy
from os.path import join


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


def find_coord_indices(lon_array, lat_array, lon_points, lat_points):
    """
    Find indices of nearest lon/lat pair on a grid. Supports rectilinear and curilinear grids.
    lon_points / lat_points must be received as a list.
    Args:
        lon_array (np.array): Array of longitude from model grid
        lat_array (np.array): Array of lattitude from model grid
        lon_points (list): List of longitude points from storm objects
        lat_points (list): List of lattitude points from storm objects

    Returns:
        List of indices (i,j) on grid where storms were located.
    """
    lonlat = np.column_stack((lon_array.ravel(), lat_array.ravel()))
    ll = np.array([lon_points, lat_points]).T
    idx = scipy.spatial.distance.cdist(lonlat, ll).argmin(0)

    return np.column_stack((np.unravel_index(idx, lon_array.shape))).tolist()


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

            storm_indxs[storm_type] = find_coord_indices(model_grid['lon'].values, model_grid['lat'].values,
                                                         df_storms[storm_type]['centroid_lon'],
                                                         df_storms[storm_type]['centroid_lat'])
            storm_grid[f'{model}_{storm_type}'] = storm_grid['lat'] * 0
            for i in storm_indxs[storm_type]:
                storm_grid[f'{model}_{storm_type}'][i[0], i[1]] = 1

    return


