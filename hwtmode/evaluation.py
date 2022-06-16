import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from os.path import join
import joblib
from hwtmode.data import transform_data, get_gmm_predictions, load_gridded_data
from hwtmode.models import load_conv_net
import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
import os
import re
from datetime import datetime
from tensorflow.keras import backend as K
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Transformer, CRS

def brier_skill_score(y_true, y_pred):
    bs_climo = np.mean((y_true.mean() - y_true) ** 2)
    bs = brier_score_loss(y_true, y_pred)
    return 1 - bs / bs_climo

def brier_score(obs, preds):
    return K.mean((preds - obs) ** 2)

def bss(obs, preds):
    bs = brier_score(obs, preds)
    obs_climo = K.mean(obs, axis=0)
    bs_climo = K.mean((obs - obs_climo) ** 2)
    bss = 1.0 - bs/(bs_climo+K.epsilon())
    return bss

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


def hazard_cond_prob(obs, preds, A, B, nprob_thresh=0.0, secondary_thresh=None):
    """
    Get conditional probability. Supports binning of neighborhood probabilities.
    Args:
        obs: SPC Observations array
        preds: HWT mode predictions array
        A: SPC Hazard ('Torn', 'Hail', 'Wind') or proxy hazard variable
        B: Condition - Storm Mode prediction ('Supercell', 'QLCS', 'Disorganized')
        nprob_thresh: Threshold of neighborhood probability (used as lower threshold when binning)
        secondary_thresh: Upper threshhold for binning (default None)
    Returns: Conditional probability: P(A|B)
    """
    if secondary_thresh is not None:
        arr = np.where((preds[f'{B}_nprob'] > nprob_thresh) & (preds[f'{B}_nprob'] <= secondary_thresh), 1, 0)
        hits = np.where((arr >= 1) & (obs[A].values >= 1), 1, 0).sum()
        mode_bin = np.where(arr >= 1, 1, 0).sum()
    else:
        hits = np.where((preds[f'{B}_nprob'] > nprob_thresh).values & (obs[A] >= 1).values, 1, 0).sum()
        mode_bin = np.where(preds[f'{B}_nprob'] > nprob_thresh, 1, 0).sum()
    cond_prob = hits / mode_bin

    return cond_prob


class Cycle(object):
    """ Cycle through set of images. Rolls between first and last. """

    def __init__(self, high_files, low_files):
        self._high_files = high_files
        self._low_files = low_files
        self._max_idx = len(high_files) - 1
        self._current_idx = 0
        self.cluster = 0

    def next(self):

        if self._current_idx == self._max_idx:
            self._current_idx = 0
        else:
            self._current_idx += 1
        return self._high_files[self._current_idx], self._low_files[self._current_idx]

    def previous(self):

        if self._current_idx == 0:
            self._current_idx = self._max_idx
        else:
            self._current_idx -= 1
        return self._high_files[self._current_idx], self._low_files[self._current_idx]

    def return_index(self):

        return self._current_idx

    def return_cluster(self):

        x = self._high_files[self._current_idx].split('/')[-1]
        print(x)
        return int(''.join(i for i in x if i.isdigit()))

    def __repr__(self):

        return f"Cycle({self._high_files})"


def natural_sort(l):
    """ Natural string sorter.
    Args:
        l (list): List of strings to be sorted
    """

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


def image_viewer(base_path, CNN_name, GMM_name, evaluator):
    """Create an image viewer widget to view the image of a certain format inside a directory.

    Args:
        base_path (str): Output directory.
        CNN_name (str): Name of the CNN from config
        GMM_name (str): Name of the GMM from config.
        evaluator (str): Name / ID of person performing the labeling.
    """

    global cluster_dict
    cluster_dict = dict(Supercell=[], QLCS=[], Disorganized=[])
    prev_button = widgets.Button(description="Prev", icon="backward", layout=Layout(width="50%", height="30%"))
    next_button = widgets.Button(description="Next", icon="forward", layout=Layout(width="50%", height="30%"))
    finish_button = widgets.Button(description="Finished", icon="fa-floppy-o", layout=Layout(width="80%", height="80%"))
    QLCS_button = widgets.Button(description="QLCS", layout=Layout(width="80%", height="75%"))
    supercell_button = widgets.Button(description="Supercell", layout=Layout(width="80%", height="75%"))
    disorganized_button = widgets.Button(description="Disorganized", layout=Layout(width="80%", height="75%"))

    img_dir = join(base_path, "plots", CNN_name, GMM_name)
    image_files_high = natural_sort([os.path.join(img_dir, x) for x in os.listdir(img_dir) if 'highest' in x])
    image_files_low = natural_sort([os.path.join(img_dir, x) for x in os.listdir(img_dir) if 'lowest' in x])
    images = Cycle(image_files_high, image_files_low)
    image = widgets.Image(value=open(image_files_high[0], "rb").read(), width="95%", height="100%")
    image2 = widgets.Image(value=open(image_files_low[0], "rb").read(), width="95%", height="100%")
    out = widgets.Output(layout={'border': '1px solid black'})

    def update_image(filename1, filename2):

        with open(filename1, "rb") as f1, open(filename2, "rb") as f2:
            image.value = f1.read()
            image2.value = f2.read()

    def update_widgets(filename):

        update_image(filename[0], filename[1])

    def handle_next(button):

        update_widgets(images.next())

    def handle_prev(button):

        update_widgets(images.previous())

    def check_and_replace_value(dictionary, cluster):

        for key in dictionary.keys():
            if cluster in dictionary[key]:
                dictionary[key].remove(cluster)
        return

    def assign_QLCS(cluster):

        check_and_replace_value(cluster_dict, cluster)
        cluster_dict['QLCS'].append(cluster)
        with out:
            out.clear_output()
            print(f"\nCluster {cluster} has been assigned 'QLCS'")
            print(cluster_dict)

    def assign_supercell(cluster):

        check_and_replace_value(cluster_dict, cluster)
        cluster_dict['Supercell'].append(cluster)
        with out:
            out.clear_output()
            print(f"\nCluster {cluster} has been assigned 'Supercell'")
            print(cluster_dict)

    def assign_Disorganized(cluster):

        check_and_replace_value(cluster_dict, cluster)
        cluster_dict['Disorganized'].append(cluster)
        with out:
            out.clear_output()
            print(f"\nCluster {cluster} has been assigned 'Disorganized'")
            print(cluster_dict)

    def on_sc_clicked(button):

        assign_supercell(images.return_cluster())

    def on_qlcs_clicked(button):

        assign_QLCS(images.return_cluster())

    def on_dis_clicked(button):

        assign_Disorganized(images.return_cluster())

    def write_file(button):

        time = datetime.now().strftime("%Y-%m-%d_%H%M")
        label_out_path = join(base_path, "models", CNN_name, f"{CNN_name}_{GMM_name}_labels_{evaluator}_{time}.dict")
        joblib.dump(cluster_dict, label_out_path)
        with out:
            out.clear_output()
            print(f"\nSuccessfully wrote the results to {label_out_path}")
            print(cluster_dict)

    prev_button.on_click(handle_prev)
    next_button.on_click(handle_next)
    supercell_button.on_click(on_sc_clicked)
    QLCS_button.on_click(on_qlcs_clicked)
    disorganized_button.on_click(on_dis_clicked)
    finish_button.on_click(write_file)

    app = GridspecLayout(9, 6, height='900px')
    app[0, 0] = QLCS_button
    app[0, 1] = supercell_button
    app[0, 2] = disorganized_button
    app[0, 3] = prev_button
    app[0, 4] = next_button
    app[0, 5] = finish_button
    app[1:8, 0:3] = image
    app[1:8, 3:] = image2
    app[-1, :] = out

    return app


class StaggeredGridder(object):

    def __init__(self, base_grid_path, longitude_name, latitude_name, latlonCRS, proj_str):

        self.base_grid_path = base_grid_path
        self.longitude_name = longitude_name
        self.latitude_name = latitude_name
        self.proj_str = proj_str
        self.lcc = CRS(proj_str)
        self.latlon = CRS(latlonCRS)
        self.transformer = Transformer.from_crs(self.latlon, self.lcc)
        self.inv_transformer = Transformer.from_crs(self.lcc, self.latlon)

    def load_base_grid(self):

        grid = xr.open_dataset(self.base_grid_path)
        lat, lon = grid[self.latitude_name].values, grid[self.longitude_name].values

        return lat, lon

    def transform_points(self, lats, lons):

        return self.transformer.transform(lats, lons)

    def inv_transform_points(self, lons_m, lats_m):

        return self.inv_transformer.transform(lats_m, lons_m)

    def make_staggered_point_grid(self):

        lats, lons = self.load_base_grid()
        latlon_m = self.transform_points(lats, lons)
        grid_x, grid_y = latlon_m[0], latlon_m[1]
        x_dim, y_dim = lats.shape[1], lats.shape[0]

        x_offset = np.abs((latlon_m[0][:, :-1] - latlon_m[0][:, 1:]).mean() / 2)
        y_offset = np.abs((latlon_m[1][:-1, :] - latlon_m[1][1:, :]).mean() / 2)

        grid_offset_x = grid_x - x_offset
        grid_offset_y = grid_y - y_offset
        expanded_x, expanded_y = np.zeros((y_dim + 1, x_dim + 1)), np.zeros((y_dim + 1, x_dim + 1))
        expanded_x[:-1, :-1], expanded_y[:-1, :-1] = grid_offset_x, grid_offset_y

        expanded_x[-1, :-1] = expanded_x[-2, :-1]
        expanded_x[:, -1] = expanded_x[:, -2] + (x_offset * 2)

        expanded_y[-1, :-1] = expanded_y[-2, :-1] + (y_offset * 2)
        expanded_y[:, -1] = expanded_y[:, -2]

        latlon = self.inv_transform_points(expanded_y, expanded_x)

        return latlon[0], latlon[1]

    def create_geo_df(self):

        lats, lons = self.make_staggered_point_grid()

        x_dim, y_dim = lats.shape[1], lats.shape[0]
        polygons = []

        for yi in range(y_dim - 1):
            for xi in range(x_dim - 1):
                polygon = []
                polygon.append((lons[yi, xi], lats[yi, xi]))
                polygon.append((lons[yi, xi + 1], lats[yi, xi + 1]))
                polygon.append((lons[yi + 1, xi + 1], lats[yi + 1, xi + 1]))
                polygon.append((lons[yi + 1, xi], lats[yi + 1, xi]))
                polygon.append((lons[yi, xi], lats[yi, xi]))
                polygons.append(Polygon(polygon))

        return gpd.GeoDataFrame(geometry=polygons)

    def save_geoJSON(self, gdf, path):

        gdf.to_file(path, driver="GeoJSON")