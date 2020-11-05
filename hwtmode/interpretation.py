import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from os.path import join
from scipy.ndimage import gaussian_filter
import pandas as pd

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

def cape_shear_modes(data_path, neuron_activations, mode='train', model_name='mod', number_neurons=4, num_storms=50, use_mask=False):
        """
        Match specified number of top storms of each neuron, fetch storm patch, and then plot bivariate density of each nueron in CAPE/Shear space. 

        Args:
            data_path: Absolute path of netcdf patch data
            nueron_activations: CSV file of neuron activations 
            mode: data partition: 'train', 'val', or 'test'
            model_name: name of model used for training
            number_neurons: number of nuerons in activation file
            num_storms: number of top activated storms to use for density estimation for each neuron

        Returns:
            df: pandas dataframe of top storm values for CAPE and Shear, split by neuron
            : bivariate density estimation plot
        """
    df = pd.DataFrame(columns=['CAPE', '6km Shear', 'Neuron'])
    for n in list(neuron_activations.columns[-number_neurons:]):
        var = ['MLCAPE_prev', 'USHR6_prev', 'VSHR6_prev']
        sub = neuron_activations.sort_values(by=[n], ascending=False).iloc[:num_storms, :].reset_index(drop=True)
        dates = sub['run_date']
        file_strings = [f'{data_path}NCARSTORM_{x.strftime("%Y%m%d")}-0000_d01_model_patches.nc' for x in dates]
        df_vals = []
        
        for i, file in enumerate(file_strings):
            ds = xr.open_dataset(file)
            x = ds[var].where((ds.centroid_i == sub['centroid_i'][i])&(ds.centroid_j == sub['centroid_j'][i]), drop=True)
            cape = x['MLCAPE_prev'].max().values
            shear = np.sqrt(x['USHR6_prev']**2 + x['VSHR6_prev']**2).mean().values
            df_vals.append([cape, shear, n])
        df = df.append(pd.DataFrame(df_vals, columns=df.columns))
        df[['CAPE','6km Shear']] = df[['CAPE','6km Shear']].astype('float32')
        
    plt.figure(figsize=(20,16))
    sns.kdeplot(data=df, x='CAPE', y='6km Shear', hue='Neuron', fill=True, alpha=0.5, thresh=0.4)
        
    return df 
