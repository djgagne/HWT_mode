import xarray as xr
import numpy as np
import os
from os import makedirs
from os.path import exists, join, isfile
from glob import glob
from tqdm import tqdm
import pandas as pd
from skimage import measure
from shapely.geometry import Polygon
import joblib
import s3fs
from sklearn.preprocessing import StandardScaler


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
        raise FileNotFoundError("No patch files found in " + patch_dir)
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
            if mask:
                masked = ds[input_variables].where(ds["masks"] > 0, 0)
                input_data_list.append(masked.compute())
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



def decompose_circular_feature(df: pd.DataFrame, *features, period=2*np.pi, drop=True):
    import logging
    """
    Decompose a circular feature like azimuth, LST, or orientation into its 1) sine and 2) cosine components.
    Drop original circular feature, unless optional drop argument is set explicitly to False.

    Args:
        df: :class:`pandas.DataFrame` being manipulated
        features: feature name(s)
        period (optional): period of cycle (default 2*np.pi)
        drop (optional): if True, drop original feature(s)

    Returns:
        df: pandas.DataFrame with sine and cosine components of feature(s) 
    """
    for feature in features:
        logging.info(f"{feature} sin and cos components, period={period}")
        df[feature+"_sin"] = np.sin(df[feature] * 2*np.pi/period)
        df[feature+"_cos"] = np.cos(df[feature] * 2*np.pi/period)
        if drop:
            logging.debug(f"drop {feature} column from dataframe")
            df = df.drop(columns=feature)
    return df


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


def get_storm_variables(start, end, data_path, csv_prefix, storm_vars):
    """
    Args:
        start: Start date (format: YYYY-MM-DD)
        end: End Date (format YYYY-MM-DD)
        data_path: Base path to data
        csv_prefix: Prefix of CSV files
        storm_vars: Variables to pull

    Returns: Numpy array (samples, variables)

    """
    start_str = (pd.Timestamp(start, tz="UTC")).strftime("%Y%m%d-%H00")
    end_str = (pd.Timestamp(end, tz="UTC")).strftime("%Y%m%d-%H00")
    l = []
    for d in pd.date_range(start_str.replace('-', ''), end_str.replace('-', ''), freq='d'):
        file_path = join(data_path, f'{csv_prefix}{d.strftime("%Y%m%d-%H00")}.csv')
        print(file_path)
        if exists(file_path):
            df = pd.read_csv(file_path)
            l.append(df)
    storm_data = pd.concat(l).reset_index(drop=True)
    storm_vals = storm_data[storm_vars].values

    return storm_vals


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
        scale_values = pd.DataFrame(0.0, index=patch_data["var_name"].values, columns=["min", "max"])
        fit = True
    transformed = patch_data.copy(deep=True)
    for v, var_name in enumerate(patch_data["var_name"].values):
        print(var_name)
        if fit:
            scale_values.loc[var_name, "min"] = float(patch_data[..., v].min().values)
            scale_values.loc[var_name, "max"] = float(patch_data[..., v].max().values)
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


def uvmagnitude(df: pd.DataFrame, drop=True):
    import logging
    """
    Look for U/V component pairs, derive magnitude.

    Args:
        df: Pandas DataFrame

    Returns:
        df: Pandas DataFrame with magnitudes
    """

    possible_components = [f'SHR{z}{potential}{sfx}' for z in "136" for potential in ["","-potential"] for sfx in ["_min","_mean","_max"]]
    possible_components += [f'10{potential}{sfx}' for potential in ["","-potential"] for sfx in ["_min","_mean","_max"]]
    logging.debug(f"uvmagnitude: possible_components={possible_components}")
    # process possible u/v components
    for possible_component in possible_components:
        uc = "U"+possible_component 
        vc = "V"+possible_component 
        if uc in df.columns and vc in df.columns:
            logging.info(f"calculate {possible_component} magnitude")
            df[possible_component] = ( df[uc]**2 + df[vc]**2 )**0.5
            if drop:
                logging.debug(f"drop {[uc,vc]} columns from dataframe")
                df = df.drop(columns=[uc,vc])
    return df



def predict_labels_gmm(neuron_acts, gmm_model, model_name, cluster_dict):
    """
    Given neuron activations, feed to GMM to produce labels and probabilities.
    Args:
        neuron_acts: Pandas dataframe of neuron activations (including meta data)
        gmm_model: Trained Gaussian Mixture Model object
        model_name: (str)
        cluster_dict: Dictionary mapping cluster numbers to storm mode

    Returns:
        Pandas DataFrame of predictions and probabilities
    """

    clust_prob_labels = [f'{model_name}_cluster_{x}_prob' for x in range(gmm_model.n_components)]
    mode_prob_labels = [f'{model_name}_{x}_prob' for x in ['QLCS', 'Supercell', 'Disorganized']]
    max_prob_label = f'{model_name}_label_prob'
    label = f'{model_name}_label'
    cols = clust_prob_labels + mode_prob_labels + [max_prob_label] + [label]

    df = pd.DataFrame(index=range(len(neuron_acts)), columns=cols, dtype='float32')
    df.loc[:, clust_prob_labels] = gmm_model.predict_proba(
        neuron_acts.loc[:, neuron_acts.columns.str.contains('neuron_')])
    for key in cluster_dict.keys():
        df[f'{model_name}_{key}_prob'] = df[[f'{model_name}_cluster_{x}_prob' for x in cluster_dict[key]]].sum(axis=1)
    df.loc[:, max_prob_label] = df.loc[:, mode_prob_labels].max(axis=1)
    max_mode = df.loc[:, mode_prob_labels].idxmax(axis=1)
    df.loc[:, label] = max_mode.apply(lambda x: x.split('_')[-2])

    return df


def predict_labels_cnn(input_data, model, model_name):
    """
    Generate labels and probabilities from CNN and add to labels
    Args:
        input_data: Scaled input data formatted for input into CNN
        model: Convolutional Neural Network (CNN) Model
        model_name: (str)
    Returns:
        Dataframe of CNN labels
    """

    mode_prob_labels = [f'{model_name}_{x}_prob' for x in ['QLCS', 'Supercell', 'Disorganized']]
    max_prob_label = f'{model_name}_label_prob'
    label = f'{model_name}_label'
    cols = mode_prob_labels + [max_prob_label] + [label]
    df = pd.DataFrame(index=range(len(input_data)), columns=cols, dtype='float32')
    df.loc[:, mode_prob_labels] = model.predict(input_data)
    df.loc[:, max_prob_label] = df.loc[:, mode_prob_labels].max(axis=1)
    max_mode = df.loc[:, mode_prob_labels].idxmax(axis=1)
    df.loc[:, label] = max_mode.apply(lambda x: x.split('_')[-2])

    return df


def predict_labels_dnn(input_data, scale_values, model, input_vars, model_name):
    """ Generate labels and probabilities from DNN and add to labels
     Args:
        input_data (df): Hagelslag csv output
        scale_values (dict): Dictionary of Column names and scale values for StandardScaler()
        model: Loaded Tensorflow model
        input_vars (list): input variables to model
        model_name: (str)
    Returns:
        Dataframe of labels, probabilities, and meta data.
     """
    scaler = StandardScaler()
    scaler.mean_ = scale_values['mean']
    scaler.scale_ = scale_values['std']

    mode_prob_labels = [f'{model_name}_{x}_prob' for x in ['QLCS', 'Supercell', 'Disorganized']]
    max_prob_label = f'{model_name}_label_prob'
    label = f'{model_name}_label'
    cols = mode_prob_labels + [max_prob_label] + [label]
    df = pd.DataFrame(index=range(len(input_data)), columns=cols, dtype='float32')
    df.loc[:, mode_prob_labels] = model.predict(scaler.transform(input_data[input_vars]))
    df.loc[:, max_prob_label] = df.loc[:, mode_prob_labels].max(axis=1)
    max_mode = df.loc[:, mode_prob_labels].idxmax(axis=1)
    df.loc[:, label] = max_mode.apply(lambda x: x.split('_')[-2])

    return df


def merge_labels(labeled_data, storm_data, meta_vars, storm_vars):
    """

    Args:
        labeled_data: Dictionary of pandas dataframes of labeled output from models
        storm_data: Dataframe of tabular output from Hagelslag
        meta_vars: List of meta variables to merge
        storm_vars: Additional storm variables from hagelslag to keep

    Returns:
        Merged dataframe with labels, meta, and storm variables
    """
    labels = pd.concat([df for df in labeled_data.values()], axis=1)
    meta_df = storm_data.loc[:, meta_vars]
    storm_df = storm_data.loc[:, storm_vars]
    all_data = pd.concat([meta_df, storm_df, labels], axis=1)

    return all_data.astype({x: 'datetime64' for x in meta_vars if 'Date' in x})


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


def load_labels(start, end, label_path, run_freq, file_format):
    """
    Load prediction files
    Args:
        start: format "YYYY-MM-DD"
        end: format "YYY-MM-DD"
        label_path: path to prediction data
        run_freq: How often to look for files to load ("daily" or "hourly")
        file_format: file format to load ("parquet", or "csv")

    Returns:
        single object with merged dataframes
    """

    labels = []
    for run_date in pd.date_range(start, end, freq=run_freq[0]):
        file_name = join(label_path, f'model_labels_{run_date.strftime("%Y-%m-%d_%H%M")}.{file_format}')
        if isfile(file_name):
            if file_format == 'parquet':
                labels.append(pd.read_parquet(file_name))
            elif file_format == 'csv':
                labels.append(pd.read_csv(file_name))
        else:
            continue

    return pd.concat(labels)


def save_labels(labels, out_path, file_format):
    """
    Save storm mode labels to parquet or csv file.
    Args:
        labels: Pandas dataframe of storm mode labels
        out_path: Path to save labels to
        file_format: File format (accepts 'csv' or 'parquet')
    Returns:
    """
    for date in pd.date_range(labels["Run_Date"].min(), labels["Run_Date"].max(), freq='h'):
        df_sub = labels.loc[labels['Run_Date'] == date]
        if len(df_sub) == 0:
            continue
        date_str = date.strftime("%Y-%m-%d_%H%M")
        file_name = join(out_path, f'model_labels_{date_str}.{file_format}')
        if file_format == 'csv':
            df_sub.to_csv(file_name, index=False)
        elif file_format == 'parquet':
            df_sub.to_parquet(file_name)
        else:
            raise ValueError(f'File format {file_format} not found. Please use "parquet" or "csv"')
        print(f'Wrote {file_name}.')


def save_gridded_labels(ds, base_path, tabular_format='csv'):

    print("Writing out probabilities...")
    run_date_str = pd.to_datetime(ds['init_time'].values).strftime('%Y%m%d%H00')
    for run_date in run_date_str:
        makedirs(join(base_path, run_date), exist_ok=True)

    for i in range(ds.time.size):

        data = ds.isel(time=[i])
        run_date = pd.to_datetime(data['init_time'].values[0]).strftime('%Y%m%d%H00')
        fh = data['forecast_hour'].values[0]
        file_str = join(base_path, run_date, f"label_probabilities_{run_date}_fh_{fh:02d}.nc")
        data.to_netcdf(file_str)
        print("Succesfully wrote:", file_str)
        data_tabular = data.to_dataframe(dim_order=('time', 'y', 'x'))
        tabular_file_str = join(base_path, run_date, f"label_probabilities_{run_date}_fh_{fh:02d}.{tabular_format}")
        if tabular_format == "csv":
            data_tabular.to_csv(tabular_file_str, index=False)
        elif tabular_format == "parquet":
            data_tabular.to_parquet(tabular_file_str)
        print("Succesfully wrote:", tabular_file_str)
    return


def save_gridded_reports(data, out_path):
    """
    Args:
        data: netCDF of storm reports
        out_path: Base path to save
    Returns:
    """
    data.to_netcdf('combined_storm_reports.nc')
    for valid_i, valid_time in enumerate(sorted(np.unique(data['valid_time'].values))):
        time = pd.Timestamp(valid_time).strftime("%Y-%m-%d_%H%M")
        ds = data.isel(Time=valid_i)
        ds.to_netcdf(join(out_path, f'storm_reports_{time}.nc'))
    return


def load_probabilities(start, end, eval_path, run_freq, file_format):
    """
    Load evaluation files
    Args:
        start: format "YYYY-MM-DD"
        end: format "YYY-MM-DD"
        eval_path: path to evaluation data
        run_freq: How often to look for files to load ("daily" or "hourly")
        file_format: file format to load ("nc", "parquet", or "csv")

    Returns:
        single object with merged dataframes or netCDF files
    """
    files = []
    for run_date in pd.date_range(start, end, freq=run_freq[0]).strftime("%Y%m%d%H%M"):
        file_names = sorted(glob(join(eval_path, run_date, f'label_probabilities_{run_date}*.{file_format}')))
        for file in file_names:
            files.append(file)

    if file_format == 'csv':

        file_dfs = [pd.read_csv(f) for f in files]

        return pd.concat(file_dfs)

    elif file_format == 'parquet':
        file_dfs = [pd.read_parquet(f) for f in files]

        return pd.concat(file_dfs)

    elif file_format == 'nc':

        return xr.open_mfdataset(files, parallel=True, concat_dim='time', combine='nested').load()
