import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join
from scipy.ndimage import gaussian_filter
import dask.dataframe as dd
import pandas as pd
import seaborn as sns
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

def cape_shear_modes(neuron_activations, output_path, data_path, model_name, mode, num_storms=5000):
    """
    Match specified number of top storms of each neuron, fetch storm patch,
    and then plot bivariate density of each nueron in CAPE/Shear space.
    Args:
        neuron_activations: CSV file of neuron activations
        output_path: Path to save output
        data_path: Absolute path of netcdf patch data
        model_name: name of model used for training
        mode: data partition: 'train', 'val', or 'test'
        num_storms: number of top activated storms to use for density estimation for each neuron

    Returns:
    """
    df = pd.DataFrame(columns=['CAPE', '6km Shear', 'Activation', 'Neuron'])
    cols = list(neuron_activations.columns[neuron_activations.columns.str.contains('neuron')])
    csv_path = data_path.rstrip('/')[:-2] + 'csv'
    
    dates = sorted(set(neuron_activations['run_date'].astype('datetime64[ns]')))
    file_strings = [join(csv_path, f'track_step_NCARSTORM_d01_{x.strftime("%Y%m%d")}-0000.csv') for x in dates]
    ddf = dd.read_csv(file_strings).compute()
    
    for neuron in cols:
        sub = neuron_activations.sort_values(by=[neuron], ascending=False).iloc[:num_storms, :]
        activation = sub[neuron].values
        x = ddf.iloc[sub.index, :]
        cape = x['MLCAPE-potential_max'].values
        shear = np.sqrt(x['USHR6-potential_mean']**2 + x['USHR6-potential_mean']**2).values
        df = df.append(pd.DataFrame(zip(cape, shear, activation, [neuron] * num_storms), columns=df.columns))
        
    plt.figure(figsize=(20, 16))
    sns.set(font_scale=1.5)
    colors = sns.color_palette("deep", len(cols))

    sns.scatterplot(data=df, x='CAPE', y='6km Shear', hue='Neuron', alpha=0.5, size='Activation', sizes=(1, 200),
                    palette=colors, edgecolors='k', linewidth=1)
    sns.kdeplot(data=df, x='CAPE', y='6km Shear', hue='Neuron', fill=False, alpha=1, thresh=0.4, levels=3,
                palette=colors, clip=(0, 6000), linewidths=8, legend=False)
    plt.title(f'Storm Activations for Top {num_storms} Storms ({mode})')
    plt.savefig(join(output_path, f'CAPE_Shear_{model_name}_{mode}.png'), bbox_inches='tight')
    
    return

def spatial_neuron_activations(neuron_activations, output_path, model_name, mode, quant_thresh=0.9):
    """
    Plot spatial distribution of top activated storms for each neuron
    Args:
        neuron_activations: CSV file of neuron activations
        output_path: Output path from config
        model_name: Model name from config
        mode: Data partition (train, va, or test)
        quant_thresh: Quantile to select storms that exceed threshold

    Returns:
    """
    fig = plt.figure(figsize=(20, 16))
    lcc = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(38.5, 38.5))
    ax = fig.add_subplot(1, 1, 1, projection=lcc)
    ax.set_extent([-120, -74, 25, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.STATES)

    neurons = list(neuron_activations.columns[neuron_activations.columns.str.contains('neuron')])
    colors = sns.color_palette("deep", len(neurons))

    for i, neuron in enumerate(neurons):
        data = neuron_activations[neuron_activations[neuron] > neuron_activations[neuron].quantile(quant_thresh)]
        var = data[neuron]
        plt.scatter(data['centroid_lon'], data['centroid_lat'], transform=ccrs.PlateCarree(), label=None,
                    color=colors[i], alpha=0.25, s=2.5)
        sns.kdeplot(data['centroid_lon'], data['centroid_lat'], data=var, levels=3, transform=ccrs.PlateCarree(),
                    linewidths=5, thresh=0, color=colors[i], linestyles='-',
                    label=f'Neuron {i}', cummulative=True)
        plt.legend(prop={'size': 16})
    plt.title(f'Storm Activations Above {quant_thresh} Quantile - {mode}', fontsize=30)
    plt.savefig(join(output_path, f'Spatial_activations_{model_name}_{mode}.png'), bbox_inches='tight')


def diurnal_neuron_activations(neuron_activations, output_path, model_name, mode, quant_thresh=0.9):
    """
    Plot diurnal distribution of each neuron
    Args:
        neuron_activations: CSV file of neuron activations
        output_path: Base output path from config
        model_name: Model name from config
        mode: Data partition (train, val, test)
        quant_thresh: Quantile to select storms that exceed threshold

    Returns:
    """
    fig, ax = plt.subplots(figsize=(20, 8))

    df = neuron_activations.copy()
    df.time = df.time.astype('datetime64[ns]').reset_index(drop=True) - pd.Timedelta(6, 'H')
    neurons = list(neuron_activations.columns[neuron_activations.columns.str.contains('neuron')])
    colors = sns.color_palette("deep", len(neurons))

    for i, neuron in enumerate(neurons):
        data = df[df[neuron] > df[neuron].quantile(quant_thresh)].groupby(df['time'].dt.hour)[neuron].count()
        plt.plot(data, linewidth=4, alpha=1, label=neuron, color=colors[i])
    plt.legend(prop={'size': 16})
    plt.title(f'Diurnal Distribution of Storm Activations Above {quant_thresh} Quantile - {mode}', fontsize=30)
    ax.set_ylabel('Number of Storms', fontsize=20)
    ax.set_xlabel('UTC - 6', fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    plt.savefig(join(output_path, f'Diurnal_activations_{model_name}_{mode}.png'), bbox_inches='tight')
