import xarray as xr
import numpy as np
from os.path import exists, join
from glob import glob
from tqdm import tqdm
import pandas as pd
from skimage import measure
from shapely.geometry import Polygon
import joblib
import s3fs
from hwtmode.models import load_conv_net



def load_patch_files(start_date: str, end_date: str, run_freq: str, patch_dir: str, input_variables: list,
                     output_variables: list, meta_variables: list,
                     patch_radius=None, mask=False) -> tuple:
    """
    Iterate through all patch files and load the input and output variables into separate py:class:`xarray.Dataset`
    objects.

    Args:
        start_date (str): Beginning of date range for file loading.
        end_date (str): End of data range for file loading.
        run_freq (str): Frequency at which to grab files
        patch_dir (str): Path to directory containing patch netCDF files.
        input_variables (list): List of input variable names.
        output_variables (list): List of output variable names.
        meta_variables (list): List of metadata variables.
        patch_radius (int): Number of grid cells on either side of center to include.
        mask (bool): True if input variables should be masked

    Returns:
        input_data (:class:`xarray.Dataset`): Datasetcontaining input variables as separate
    """
    if not exists(patch_dir):
        raise FileNotFoundError(f"Patch directory {patch_dir} not found")
    patch_files = pd.Series(sorted(glob(join(patch_dir, "*.nc"))))
    date_strings = patch_files.str.split("/").str[-1].str.split("_").str[1]
    patch_dates = pd.to_datetime(date_strings)
    if start_date == "today":
        if run_freq == "hourly":
            start_date_stamp = pd.Timestamp(pd.Timestamp(start_date, tz="UTC").strftime("%Y-%m-%d %H:00:00")
                                            ) - pd.Timedelta(hours=3)
            end_date_stamp = pd.Timestamp(pd.Timestamp(end_date, tz="UTC").strftime("%Y-%m-%d %H:00:00")
                                          ) - pd.Timedelta(hours=3)
        elif run_freq == 'daily':
            start_date_stamp = pd.Timestamp(pd.Timestamp(start_date, tz="UTC").strftime("%Y-%m-%d 00:00:00"))
            end_date_stamp = pd.Timestamp(pd.Timestamp(end_date, tz="UTC").strftime("%Y-%m-%d 00:00:00"))

    else:
        start_date_stamp = pd.Timestamp(pd.Timestamp(start_date, tz="UTC").strftime("%Y-%m-%d %H:00:00"))
        end_date_stamp = pd.Timestamp(pd.Timestamp(end_date, tz="UTC").strftime("%Y-%m-%d %H:00:00"))
    date_filter = (patch_dates >= start_date_stamp) & (patch_dates <= end_date_stamp)
    valid_patch_files = patch_files[date_filter]
    valid_patch_dates = patch_dates[date_filter]
    if len(valid_patch_files) == 0:
        raise FileNotFoundError("No patch files found in " + patch_1dir)
    input_data_list = []
    output_data_list = []
    meta_data_list = []
    for p, patch_file in enumerate(tqdm(valid_patch_files, ncols=60)):
        ds = xr.open_dataset(patch_file)
        ds["run_date"] = xr.DataArray([valid_patch_dates.values[p]] * ds.dims["p"],
                                         dims=("p",), name="run_date")
        if patch_radius is not None:
            row_mid = int(np.round(((ds["row"].max() - ds["row"].min()) / 2).values[()]))
            col_mid = int(np.round(((ds["col"].max() - ds["col"].min()) / 2).values[()]))
            row_slice = slice(row_mid - patch_radius, row_mid + patch_radius)
            col_slice = slice(col_mid - patch_radius, col_mid + patch_radius)
            if mask:
                masked = ds[input_variables].where(ds["masks"] > 0, 0)
                input_data_list.append(masked.sel(row=row_slice, col=col_slice).compute())
            else:
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
    input_data["p"] = np.arange(input_data["p"].size)
    output_data["p"] = np.arange(output_data["p"].size)
    meta_data["p"] = np.arange(meta_data["p"].size)
    return input_data, output_data, meta_data

def get_meta_scalars(meta_data):
    meta_vars = list(meta_data.data_vars.keys())
    scalar_vars = []
    for meta_var in meta_vars:
        if meta_data[meta_var].dims == ("p",):
            scalar_vars.append(meta_var)
    print(scalar_vars)
    return meta_data[scalar_vars].to_dataframe()


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
        patch_data: Input data arranged in (p, rol, col, var_name) dimensions
        scale_values: pandas.DataFrame containing min and max values for each variable.
    Returns:
        transformed: patch_data rescaled from 0 to 1
    """
    fit = False
    if scale_values is None:
        scale_values = pd.DataFrame(0, index=patch_data["var_name"].values, columns=["min", "max"])
        fit = True
    transformed = patch_data.copy(deep=True)
    for v, var_name in enumerate(patch_data["var_name"].values):
        print(var_name)
        if fit:
            scale_values.loc[var_name, "min"] = patch_data[..., v].min().values
            scale_values.loc[var_name, "max"] = patch_data[..., v].max().values
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


def storm_max_value(output_data: xr.DataArray, masks: xr.DataArray) -> np.ndarray:
    """


    Args:
        output_data:
        masks:

    Returns:

    """
    max_values = (output_data * masks).max(axis=-1).max(axis=-1).values
    return max_values


def predict_labels_gmm(neuron_acts, neuron_columns, gmm_model, cluster_dict, objects=True):
    """
    Given neuron activations, feed to GMM to produce labels and probabilities.
    Args:
        neuron_acts: Pandas dataframe of neuron activations (including meta data)
        neuron_columns: list of columns containing the activations
        gmm_model: Trained Gaussian Mixture Model object
        cluster_dict: Dictionary mapping cluster numbers to storm mode

    Returns:
        Pandas DataFrame of predictions and probabilities (including meta data)
    """

    prob_labels = [f'cluster_{x}_prob' for x in range(gmm_model.n_components)]
    neuron_acts['label'] = -9999
    neuron_acts['cluster'] = gmm_model.predict(neuron_acts.loc[:, neuron_columns])
    neuron_acts[prob_labels] = gmm_model.predict_proba(
        neuron_acts.loc[:, neuron_acts.columns.str.contains('neuron_')])

    for key in cluster_dict.keys():
        neuron_acts.loc[neuron_acts['cluster'].isin(cluster_dict[key]), 'label'] = key
        neuron_acts[f'{key}_prob'] = neuron_acts[[f'cluster_{x}_prob' for x in cluster_dict[key]]].sum(axis=1)
        neuron_acts[key] = 0
        neuron_acts.loc[neuron_acts['label'].isin([key]), key] = 1
        labels_w_meta = neuron_acts.loc[:, ~neuron_acts.columns.isin(neuron_columns)]

    labels_w_meta.loc[:, 'label_int'] = labels_w_meta['label'].factorize()[0]
    labels_w_meta.loc[:, 'label_prob'] = labels_w_meta[['Supercell_prob', 'QLCS_prob', 'Disorganized_prob']].max(axis=1)
    if objects:
        labels_w_meta.insert(1, 'forecast_hour', ((labels_w_meta['time'] - labels_w_meta['run_date']) /
                                                  pd.Timedelta(hours=1)).astype('int32'))


    return labels_w_meta


def predict_labels_cnn(input_data, meta_df, model, objects=True):
    """
    Generate labels and probabilities from CNN and add to labels
    Args:
        input_data: Scaled input data formatted for input into CNN
        geometry: Dataframe of storm patch geometry (including meta data)
        model: Convolutional Neural Network (CNN) Model
    Returns:
        Dataframe with appended new CNN labels
    """
    df = meta_df.copy()
    preds = model.predict(input_data)
    df['label'] = -9999
    df['label_int'] = preds.argmax(axis=1)
    df['label_prob'] = preds.max(axis=1)
    for i, label in enumerate(['QLCS', 'Supercell', 'Disorganized']):
        df[label] = 0
        df[f'{label}_prob'] = preds[:, i]
        df.loc[df['label_int'] == i, 'label'] = label
        df.loc[df['label_int'] == i, label] = 1
    if objects:
        df.insert(1, 'forecast_hour', ((df['time'] - df['run_date']) / pd.Timedelta(hours=1)).astype('int32'))
    return df


def lon_to_web_mercator(lon):
    """
    Transform longitudes to web_mercator projection in meters
    Args:
        lon: longitude

    Returns:
        web_mercator transformation in meters
    """
    k = 6378137
    return lon * (k * np.pi / 180.0)


def lat_to_web_mercator(lat):
    """
        Transform latitudes to web_mercator projection in meters
        Args:
            lat: latitude

        Returns:
            web_mercator transformation in meters
        """
    k = 6378137
    return np.log(np.tan((90 + lat) * np.pi / 360.0)) * k


def get_xy_coords(storms):
    """
    Takes Polygons of storm masks as paired coordinates and returns seperated x and y coordinates
    Args:
        storms: List of polygon storms [x, y]

    Returns:
        x: list of x coordinates
        y: list of y coordinates
    """
    x, y = [], []
    [(x.append(list(polygon.exterior.coords.xy[0])), y.append(list(polygon.exterior.coords.xy[1]))) for polygon in
     storms]

    return x, y


def get_contours(data):
    """
    Takes storm masks (netCDF) and generates storm outlines in lat-lon coordinates
    Args:
        data: netCDF file of storm masks with meta data

    Returns:

    """

    masks = data["masks"].values.astype(np.float32)
    lons = data.variables["lon"].values.astype(np.float32)
    lats = data.variables["lat"].values.astype(np.float32)

    storms = []
    storms_lcc = []
    skips = []
    for i, mask in enumerate(masks):
        contours = measure.find_contours(mask, 0.01)[0]
        lons_m = []
        lats_m = []
        lats_list, lons_list = [], []
        for contour in np.round(contours).astype(np.int32):
            row = contour[0]
            col = contour[1]
            lons_m.append(lon_to_web_mercator(lons[i][row, col]))
            lats_m.append(lat_to_web_mercator(lats[i][row, col]))
            lons_list.append(lons[i][row, col])
            lats_list.append(lats[i][row, col])
        try:
            storms.append(Polygon(list(zip(lons_m, lats_m))))
            storms_lcc.append(Polygon(list(zip(lons_list, lats_list))))
        except:
            print(f"Storm {i} doesn't have enough points {list(zip(lons_m, lats_m))} to create Polygon")
            skips.append(i)
    print('Generating mask outlines...')
    x, y = get_xy_coords(storms)
    lon, lat = get_xy_coords(storms_lcc)

    data = data.to_dataframe()
    data = data.reset_index(level=[0, 1, 2]).drop_duplicates(subset='p', keep='first')
    data = data.drop(['p', 'i', 'j', 'col', 'masks', 'row', 'lat', 'lon'], axis=1).reset_index(drop=True)
    data = data.drop(skips)
    data["x"] = x
    data["y"] = y
    data['lat'] = lat
    data['lon'] = lon

    return data, skips


def load_wrf_patches(start_date, end_date, output_dir, input_vars, output_vars, meta_vars, patch_radius):
    dates = pd.date_range(start_date, end_date)
    patch_size = patch_radius * 2 + 1
    all_files = []
    for date in dates:
        date = date.strftime('%Y%m%d00')
        file_list = sorted(glob(join(output_dir, date, 'wrf_rundir', 'ens_1', 'diags*')))
        all_files.append(file_list)
    all_files = [i for sub in all_files for i in sub]
    p_list = []
    for file in all_files:
        d = xr.open_dataset(file)
        ns, we, = d['south_north'].size, d['west_east'].size
        for i in np.arange(0, ns - patch_size, patch_size):
            for j in np.arange(0, we - patch_size, patch_size):
                p = d[meta_vars + input_vars + output_vars].isel(south_north=slice(i, i + patch_size),
                                                                 west_east=slice(j, j + patch_size)).squeeze()
                if p['REFL_COM'].values.max() > 35:
                    p_list.append(p)
    ds = xr.concat(p_list, dim='p').rename_dims({'south_north': 'row', 'west_east': 'col'})
    input_data = ds[input_vars]
    output_data = ds[output_vars]
    meta = ds[meta_vars]
    return input_data, output_data, meta


def get_gmm_predictions(patch, cnn_mod, model_path, model_name):
    """
    Generate predictions from semi-supervised CNN/GMM model pipeline.
    Args:
        patch: Patch to be fed into CNN model for neuron activations
        cnn_mod: CNN model object
        model_path: Base path for models
        mdoel_name: Model name
    Returns:
        List of GMM probabilities for patch
    """
    gmm_mod = joblib.load(join(model_path, model_name, f'{model_name}.gmm'))
    cluster_assignments = joblib.load(join(model_path, model_name, f'{model_name}_gmm_labels.dict'))
    neuron_activations = cnn_mod.output_hidden_layer(patch)
    preds = gmm_mod.predict_proba(neuron_activations)
    pred_list = []
    for mode_type in ['QLCS', 'Supercell', 'Disorganized']:
        mode_prediction = preds[0][cluster_assignments[mode_type]].sum()
        pred_list.append(mode_prediction)
    return [pred_list]


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
    ds_coarse['QLCS'].values = np.where(ds_coarse['QLCS_prob'] > (1 / 3), 1, np.nan)
    ds_coarse['Supercell'].values = np.where(ds_coarse['Supercell_prob'] > (1 / 3), 1, np.nan)
    ds_coarse['Disorganized'].values = np.where(ds_coarse['Disorganized_prob'] > (1 / 3), 1, np.nan)
    daily_list.append(ds_coarse)
    del ds_coarse

    return daily_list


def transform_data(patch_data, variables, scale_values):
    """
    Combines separate DataArrays from a Dataset into one combined and scaled DataArray for input to deep learning.

    Args:
        patch_data: :class:`xarray.Dataset` being combined.
        variables: List of variable names.
        scale_values: Dataframe of Min/Max Scale values for each variable

    Returns:
         sclaed_data: Scaled xarray.DataArray with dimensions (p, row, col, var_name)
    """
    combined = xr.concat([patch_data[variable] for variable in variables],
                         pd.Index(variables, name="var_name")).transpose("Time", "row", "col", "var_name")
    scaled_data, scale_v = min_max_scale(combined, scale_values)
    return scaled_data


def load_gridded_data(data_path, physical_model, run_date, forecast_hour, input_vars, output_vars, meta_vars):
    """
    Load Model data across entire grid for specified model run. Supports WRF and HRRR.

    Args:
        data_path: Base directory for Raw WRF output or AWS S3 bucket location for HRRR data.
        physical_model: Model used to generate data. Supports 'HRRR' or 'WRF'.
        run_date: model run date in format YYYYMMDD (or "today")
        forecast_hour: Model forecast Hour in format HH
        input_vars: Input variables. Must match those for model data was generated with. For HRRR, use format {variable_name}-{name_of_level}
        output_vars: Output variables.
        meta_vars: Meta variables to be kept.
    Returns:
        Xarray dataset loaded into memory
    """

    if physical_model.upper() == 'WRF':

        # ds = xr.open_mfdataset(join(data_path, run_date, forecast_hour, 'wrf_rundir', 'ens_1', 'diags*'),
        #                        combine='nested', concat_dim='Time', parallel=True)
        ds = xr.open_mfdataset(join(data_path, "diags*"), combine='nested', concat_dim='Time', parallel=True)
        ds = ds.rename_dims(dict(south_north='row', west_east='col'))

        return ds[input_vars + output_vars + meta_vars].load()

    elif physical_model.upper() == 'HRRR':

        run_date_str = pd.to_datetime(run_date).strftime("%Y%m%d")
        forecast_hour_str = pd.to_datetime(join(run_date, forecast_hour)).strftime("%H")
        datasets = []

        for variable in input_vars:
            files = []
            level = variable.split('-')[1]
            variable = variable.split('-')[0]
            fs = s3fs.S3FileSystem(anon=True)

            coord_path = join(data_path, run_date_str, f'{run_date_str}_{forecast_hour_str}z_fcst.zarr', level,
                              variable)
            f = s3fs.S3Map(root=coord_path, s3=fs, check=False)
            files.append(f)

            path = join(data_path, run_date_str, f'{run_date_str}_{forecast_hour_str}z_fcst.zarr', level, variable,
                        level)
            f = s3fs.S3Map(root=path, s3=fs, check=False)
            files.append(f)

            ds = xr.open_mfdataset(files, engine='zarr').load()
            datasets.append(ds)

        all_ds = xr.merge(datasets)
        all_ds = all_ds.rename_dims(dict(projection_x_coordinate='col', projection_y_coordinate='row', time='Time'))

        return all_ds.load()
