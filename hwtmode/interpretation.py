import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from os.path import join
from scipy.ndimage import gaussian_filter
import pandas as pd
import xarray as xr

def corr_coef_metric(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def score_neurons(y_true, neuron_activations, metric="auc"):
    scores = np.zeros(neuron_activations.shape[1], dtype=np.float32)
    if metric == "auc":
        for i in range(neuron_activations.shape[1]):
            scores[i] = roc_auc_score(y_true, neuron_activations[:, i])
    elif metric == "r":
        for i in range(neuron_activations.shape[1]):
            scores[i] = corr_coef_metric(y_true, neuron_activations[:, i])
    else:
        for i in range(neuron_activations.shape[1]):
            scores[i] = roc_auc_score(y_true, neuron_activations[:, i])
    return scores


def plot_neuron_composites(out_path, model_desc, x_data, neuron_activations, neuron_scores, variable_name,
                           composite_size=30,
                           figsize_scale=3.0, out_format="png", dpi=200, plot_kwargs=None,
                           colorbar_loc=(0.93, 0.1, 0.02, 0.8)):
    neuron_ranking = np.argsort(neuron_scores)[::-1]
    variable_index = np.where(x_data.var_name == variable_name)[0][0]
    if plot_kwargs is None:
        plot_kwargs = {}
    fig_rows = int(np.floor(np.sqrt(neuron_scores.size)))
    fig_cols = int(np.ceil(neuron_scores.size / fig_rows))
    fig, axes = plt.subplots(fig_rows, fig_cols,
                             figsize=(fig_cols * figsize_scale, fig_rows * figsize_scale))
    plot_kwargs["vmin"] = x_data[..., variable_index].min()
    plot_kwargs["vmax"] = x_data[..., variable_index].max()
    pc = None
    for a, ax in enumerate(axes.ravel()):
        if a >= neuron_scores.size:
            ax.set_visible(False)
            continue
        example_rankings = np.argsort(neuron_activations[:, neuron_ranking[a]])[::-1][:composite_size]
        x_composite = x_data[example_rankings, :, :, variable_index].mean(axis=0)
        pc = ax.pcolormesh(x_composite, **plot_kwargs)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_title("Neuron {0:d} Score {1:0.3f}".format(neuron_ranking[a],
                                                          neuron_scores[neuron_ranking[a]]))
    if pc is not None:
        cb_ax = fig.add_axes(colorbar_loc)
        cbar = fig.colorbar(pc, cax=cb_ax)
    fig.suptitle(model_desc.replace("_", " ") + " " + variable_name + " Neuron Example Composites")
    plt.savefig(join(out_path, f"neuron_composite_{variable_name}_{model_desc}.{out_format}"),
                dpi=dpi, bbox_inches="tight")
    plt.close()
    return


def plot_saliency_composites(out_path, model_name, saliency_data, neuron_activations, neuron_scores,
                             variable_name, composite_size=30, figsize_scale=3.0,
                             out_format="png", dpi=200, plot_kwargs=None,
                             colorbar_loc=(0.93, 0.1, 0.02, 0.8)):
    neuron_ranking = np.argsort(neuron_scores)[::-1]
    variable_index = np.where(saliency_data.var_name == variable_name)[0][0]
    if plot_kwargs is None:
        plot_kwargs = {}
    fig_rows = int(np.floor(np.sqrt(neuron_scores.size)))
    fig_cols = int(np.ceil(neuron_scores.size / fig_rows))
    fig, axes = plt.subplots(fig_rows, fig_cols,
                             figsize=(fig_cols * figsize_scale, fig_rows * figsize_scale))
    plot_kwargs["vmin"] = saliency_data.sel(var_name=variable_name).min().values[()]
    plot_kwargs["vmax"] = saliency_data.sel(var_name=variable_name).max().values[()]
    pc = None
    for a, ax in enumerate(axes.ravel()):
        if a >= neuron_scores.size:
            ax.set_visible(False)
            continue
        example_rankings = np.argsort(neuron_activations[:, neuron_ranking[a]])[::-1][:composite_size]
        saliency_composite = saliency_data[neuron_ranking[a], example_rankings, :, :, variable_index].mean(axis=0)
        pc = ax.pcolormesh(saliency_composite, **plot_kwargs)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_title("Neuron {0:d} Score {1:0.3f}".format(neuron_ranking[a],
                                                          neuron_scores[neuron_ranking[a]]))
    if pc is not None:
        cb_ax = fig.add_axes(colorbar_loc)
        cbar = fig.colorbar(pc, cax=cb_ax)
    fig.suptitle(model_name.replace("_", " ") + " " + variable_name + " Saliency Composites")
    plt.savefig(join(out_path, f"saliency_composite_{variable_name}_{model_name}.{out_format}"),
                dpi=dpi, bbox_inches="tight")
    plt.close()
    return


def plot_top_activations(out_path, model_name, x_data, meta_df, neuron_activations, neuron_scores, saliency_data,
                         variable_name, panel_size=16, figsize_scale=3.0, out_format="png", dpi=200, plot_kwargs=None,
                         colorbar_loc=(0.93, 0.1, 0.02, 0.8)):
    if plot_kwargs is None:
        plot_kwargs = {}
    fig_rows = int(np.floor(np.sqrt(panel_size)))
    fig_cols = int(np.ceil(panel_size / fig_rows))
    for neuron_number in range(neuron_scores.size):
        n_rank = neuron_activations[f"neuron_{neuron_number:03d}"].argsort()[::-1].values
        fig, axes = plt.subplots(fig_rows, fig_cols,
                                 figsize=(fig_cols * figsize_scale, fig_rows * figsize_scale),
                                 sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)
        sal_ex = saliency_data[neuron_number, n_rank[:panel_size]].sel(var_name=variable_name)
        sal_max = np.abs(sal_ex).max()
        pc = None
        for a, ax in enumerate(axes.ravel()):
            pc = ax.pcolormesh(x_data[n_rank[a], :, :, 0], **plot_kwargs)
            ax.contour(-sal_ex[a], 6, vmin=-sal_max, vmax=sal_max, cmap="RdBu_r")
            ax.set_xticks(np.arange(0, 32, 8))
            ax.set_yticks(np.arange(0, 32, 8))
            ex_n_score = neuron_activations.loc[n_rank[a], f"neuron_{neuron_number:03d}"]
            ax.text(0, 0, pd.Timestamp(meta_df.loc[n_rank[a], "time"]).strftime("%Y-%m-%d %HZ") + " S:{0:0.2f}".format(ex_n_score),
                    bbox=dict(facecolor='white', alpha=0.5))
        if pc is not None:
            cb_ax = fig.add_axes(colorbar_loc)
            cbar = fig.colorbar(pc, cax=cb_ax)
        fig.suptitle(f"Neuron {neuron_number} Top Activated Storms, Score: {neuron_scores[neuron_number]:0.3f}",
                     fontsize=14, y=0.95)
        plt.savefig(join(out_path, f"top_activations_neuron_{variable_name}_{neuron_number:03d}_{model_name}.{out_format}"),
                    dpi=dpi, bbox_inches="tight")
        plt.close()
    return

def plot_additional_vars(neuron_activations, data_path, output_path, mode, model_name, num_neurons, variables, plot_kwargs):
    
    neuron_activations.run_date = neuron_activations.run_date.astype('datetime64[ns]').reset_index(drop=True)
    neuron_activations.time = neuron_activations.time.astype('datetime64[ns]').reset_index(drop=True) - pd.Timedelta(6, 'H')
    
    for n in list(neuron_activations.columns[-num_neurons:]):
    
        sub = neuron_activations.sort_values(by=[n], ascending=False).iloc[:9, :].reset_index(drop=True)
        date = neuron_activations.sort_values(by=[n], ascending=False)['run_date'][:9].reset_index(drop=True)
    
        for var in variables:
            fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True)
            plt.subplots_adjust(wspace=0.03, hspace=0.03)
            kwargs = plot_kwargs[var]
            
            for i, ax in enumerate(axes.ravel()):
                date_string = date[i].strftime('%Y%m%d')
                file_path = f'{data_path}/NCARSTORM_{date_string}-0000_d01_model_patches.nc'
                ds = xr.open_dataset(file_path)
                
                if var == 'SHR6_prev':
                    u_shear = ds['USHR6_prev'].where((ds.centroid_i == sub['centroid_i'][i])&(ds.centroid_j == sub['centroid_j'][i]), drop=True)
                    v_shear = ds['VSHR6_prev'].where((ds.centroid_i == sub['centroid_i'][i])&(ds.centroid_j == sub['centroid_j'][i]), drop=True)
                    x = np.mean(np.abs(u_shear) + np.abs(v_shear), axis=0).expand_dims('p')
                else:
                    x = ds[var].where((ds.centroid_i == sub['centroid_i'][i])&(ds.centroid_j == sub['centroid_j'][i]), drop=True)
                    
                im = ax.contourf(x[0], levels=np.linspace(kwargs['vmin'], kwargs['vmax'], 101), extend='max', plot_kwargs=kwargs)
                plt.subplots_adjust(right=0.975)
                cbar_ax = fig.add_axes([1, 0.125, 0.025, 0.83])
                fig.colorbar(im, cbar_ax)
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                plt.suptitle(f'Top Storm Activations for {n} - {var} - {model_name} - {mode}', fontsize=20)
                plt.subplots_adjust(top=0.95)
                plt.savefig(f'{output_path}/{var}_{model_name}_{mode}_{n}.png', bbox_inches='tight')
    return   
