import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join, exists
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

lat_lon_proj = ccrs.PlateCarree()
lcc_proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5,
                             standard_parallels=(38.5, 38.5))
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor="none")
states_50m = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', "50m",
                                          edgecolor="k", facecolor="none")


def plot_storm_mode_analysis_map(neuron_activations, meta_data, plot_settings,
                                 run_date, start_hour, end_hour, cnn_name, out_path,
                                 period_total=False, figsize=(10, 6), transparent=False, region="CONUS"):
    neurons_plotted = sorted(list(plot_settings.keys()))
    patch_hours = ((pd.DatetimeIndex(meta_data["time"]) - pd.Timestamp(run_date)).total_seconds() // 3600).astype(int)
    run_date_str = run_date.strftime("%Y%m%d%H")
    out_full_path = join(out_path, cnn_name, run_date_str, region)
    if not exists(out_full_path):
        os.makedirs(out_full_path)
    if period_total:
        time_valid_patches = (patch_hours >= start_hour) & (patch_hours <= end_hour)
        neuron_subset = neuron_activations.loc[time_valid_patches, neurons_plotted]
        top_neuron = neuron_subset.idxmax(axis=1)
        fig, ax = plt.subplots(1, 1, figsize=figsize, projection=lcc_proj)
        ax.add_feature(land_50m)
        ax.add_feature(states_50m)
        for p in neuron_subset.index:
            n_max = neurons_plotted[top_neuron[p]]
            if neuron_subset.iloc[p, top_neuron[p]] >= plot_settings[n_max]["vmin"]:
                ax.pcolormesh(meta_data["lon"][p], meta_data["lat"][p],
                              np.ma.array(meta_data["masks"][p] * neuron_subset.iloc[p, top_neuron[p]],
                                          mask=meta_data["masks"][p] == 0),
                              vmin=plot_settings[n_max]["vmin"],
                              vmax=plot_settings[n_max]["vmax"],
                              cmap=plot_settings[n_max]["cmap"],
                              transform=lat_lon_proj)
        title_date = run_date.strftime("%Y-%m-%d %H UTC")
        ax.set_title("Convolutional Neural Network Analysis (Red: Supercell, Blue: Squall Line)\n" +
                      f"Init: {title_date} Valid: F{start_hour:03d}-F{end_hour:03d}")
        plt.savefig(join(out_full_path, f"cnn_storm_mode_total_f{start_hour:03d}-f{end_hour:03d}_{region}.png"),
                    dpi=90, bbox_inches="tight", transparent=transparent)
        plt.close()
    else:
        title_date = run_date.strftime("%Y-%m-%d %H UTC")
        for f_hour in range(start_hour, end_hour + 1):
            time_valid_patches = patch_hours == f_hour
            neuron_subset = neuron_activations.loc[time_valid_patches, neurons_plotted]
            top_neuron = neuron_subset.idxmax(axis=1)
            fig, ax = plt.subplots(1, 1, figsize=figsize, projection=lcc_proj)
            ax.add_feature(land_50m)
            ax.add_feature(states_50m)
            for p in neuron_subset.index:
                n_max = neurons_plotted[top_neuron[p]]
                if neuron_subset.iloc[p, top_neuron[p]] >= plot_settings[n_max]["vmin"]:
                    ax.pcolormesh(meta_data["lon"][p], meta_data["lat"][p],
                                  np.ma.array(meta_data["masks"][p] * neuron_subset.iloc[p, top_neuron[p]],
                                              mask=meta_data["masks"][p] == 0),
                                  vmin=plot_settings[n_max]["vmin"],
                                  vmax=plot_settings[n_max]["vmax"],
                                  cmap=plot_settings[n_max]["cmap"],
                                  transform=lat_lon_proj)
            ax.set_title("Convolutional Neural Network Analysis (Red: Supercell, Blue: Squall Line)\n" +
                         f"Init: {title_date} Valid: F{f_hour:03d}")
            plt.savefig(join(out_full_path, f"cnn_storm_mode_f{f_hour:02d}_{region}.png"),
                        dpi=90, bbox_inches="tight", transparent=transparent)
            plt.close()


    return