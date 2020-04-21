import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from os.path import join
from scipy.ndimage import gaussian_filter


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
    fig.suptitle(model_desc.replace("_", " ") + " Neuron Example Composites")
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
    fig.suptitle(model_name.replace("_", " ") + " Saliency Composites")
    plt.savefig(join(out_path, f"saliency_composite_{variable_name}_{model_name}.{out_format}"),
                dpi=dpi, bbox_inches="tight")
    plt.close()
    return
