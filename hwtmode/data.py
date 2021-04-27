import xarray as xr
import numpy as np
from os.path import exists, join
from glob import glob
from tqdm import tqdm
import pandas as pd
from skimage import measure
from shapely.geometry import Polygon

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


def storm_max_value(output_data: xr.DataArray, masks: xr.DataArray) -> np.ndarray:
    """


    Args:
        output_data:
        masks:

    Returns:

    """
    max_values = (output_data * masks).max(axis=-1).max(axis=-1).values
    return max_values


def predict_labels_gmm(neuron_acts, neuron_columns, gmm_model, cluster_dict):
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
    labels_w_meta.insert(1, 'forecast_hour', ((labels_w_meta['time'] - labels_w_meta['run_date']) /
                                              pd.Timedelta(hours=1)).astype('int32'))


    return labels_w_meta


def predict_labels_cnn(input_data, meta_df, model):
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
        contours = measure.find_contours(mask)[0]
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
    lat, lon = get_xy_coords(storms_lcc)

    data = data.to_dataframe()
    data = data.reset_index(level=[0, 1, 2]).drop_duplicates(subset='p', keep='first')
    data = data.drop(['p', 'i', 'j', 'col', 'masks', 'row', 'lat', 'lon'], axis=1).reset_index(drop=True)
    data = data.drop(skips)
    data["x"] = x
    data["y"] = y
    data['lat'] = lat
    data['lon'] = lon

    return data, skips



