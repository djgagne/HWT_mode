import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from os.path import join
import joblib
from hwtmode.data import transform_data, get_gmm_predictions, load_gridded_data
from hwtmode.models import load_conv_net


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


