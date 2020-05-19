import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join, exists
from datetime import datetime, timedelta
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import os
from mpl_toolkits.basemap import Basemap

#lat_lon_proj = ccrs.PlateCarree()
#lcc_proj = ccrs.LambertConformal(central_longitude=-101, central_latitude=38.5,
#                             standard_parallels=(32.0, 46.0))
#land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
#                                        edgecolor='k',
#                                        facecolor="none", lw=0.5)
#states_50m = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', "50m",
#                                          edgecolor="k", facecolor="0.8", lw=0.5)
#lakes_50m = cfeature.NaturalEarthFeature("physical", "lakes", "50m", edgecolor="k",
#                                         facecolor="none", lw=0.3)
bmap = Basemap(projection="lcc", llcrnrlon=-120.811, llcrnrlat=23.1593, urcrnrlon=-65.0212, 
               urcrnrlat=46.8857, area_thresh=1000, lat_1=32, lat_2=46, lon_0=-101, resolution="i")

def plot_storm_mode_analysis_map(neuron_activations, meta_data, plot_settings,
                                 run_date, start_hour, end_hour, cnn_name, out_path,
                                 period_total=False, figsize=(10, 6), dpi=90, transparent=False, region="CONUS", extent=(-120, -74, 23, 50)):
    neurons_plotted = sorted(list(plot_settings.keys()))
    patch_hours = ((pd.DatetimeIndex(meta_data["time"].values) - pd.Timestamp(run_date)).total_seconds() // 3600).astype(int)
    run_date_str = run_date.strftime("%Y%m%d%H")
    out_full_path = join(out_path, cnn_name, run_date_str, region)
    title_str = "Convolutional Neural Network Analysis (Red: Supercell, Blue: Squall Line)"
    fig_width_pixels = 1080
    fig_width  = fig_width_pixels/float(dpi)
    fig_height = fig_width * bmap.aspect + 0.93
    x,y,w,h = 0.01, 0.7/float(fig_height), 0.98, 0.98 * fig_width * bmap.aspect/float(fig_height)
    if not exists(out_full_path):
        os.makedirs(out_full_path)
    if period_total:
        time_valid_patches = (patch_hours >= start_hour) & (patch_hours <= end_hour)
        neuron_subset = neuron_activations.loc[time_valid_patches, neurons_plotted]
        top_neuron = neuron_subset.idxmax(axis=1)
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        ax = fig.add_axes([x,y,w,h])
        bmap.drawcoastlines(linewidth=0.5, ax=ax)
        bmap.drawstates(linewidth=0.25, ax=ax)
        bmap.drawcountries(ax=ax)

        #ax = fig.add_axes([0, 1, 1, 1], projection=lcc_proj)
        #ax.add_feature(land_50m, zorder=2)
        #ax.add_feature(states_50m, zorder=1)
        #ax.add_feature(lakes_50m, zorder=2)
        #ax.set_extent(extent, crs=lat_lon_proj)
        for p in neuron_subset.index:
            n_max = top_neuron[p]
            px, py = bmap(meta_data["lon"].sel(p=p).values, meta_data["lat"].sel(p=p).values)
            ax.pcolormesh(px, py,
                              np.ma.array(meta_data["masks"].sel(p=p) * neuron_subset.loc[p, n_max],
                                          mask=meta_data["masks"].sel(p=p) == 0),
                              vmin=plot_settings[n_max]["vmin"],
                              vmax=plot_settings[n_max]["vmax"],
                              cmap=plot_settings[n_max]["cmap"],
                              zorder=3)
        plot_title_time(ax, title_str, run_date.to_pydatetime(), start_hour, end_hour)
        tx, ty = ax.transAxes.transform((0,0))
        fig.figimage(plt.imread('ncar.png'), xo=tx, yo=(ty-44), zorder=1000)
        plt.text(tx+10, ty-54, 'ensemble.ucar.edu', fontdict={'size':9, 'color':'#505050'}, transform=None)
        plt.savefig(join(out_full_path, f"cnn_storm_mode_total_f{start_hour:03d}-f{end_hour:03d}_{region}.png"),
                    dpi=dpi, bbox_inches="tight", transparent=transparent)
        plt.close()
    else:
        title_date = run_date.strftime("%Y-%m-%d %H UTC")
        for f_hour in range(start_hour, end_hour + 1):
            print("plotting " + run_date_str + f" F{f_hour:03d}")
            time_valid_patches = patch_hours == f_hour
            patch_count = np.count_nonzero(time_valid_patches)

            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            ax = fig.add_axes([x,y,w,h])
            bmap.drawcoastlines(linewidth=0.5, ax=ax)
            bmap.drawstates(linewidth=0.25, ax=ax)
            bmap.drawcountries(ax=ax)
            #fig = plt.figure(figsize=figsize, dpi=dpi)
            #ax = fig.add_axes([0, 1, 1, 1], projection=lcc_proj)
            #ax.add_feature(land_50m, zorder=2)
            #ax.add_feature(states_50m, zorder=1)
            #ax.add_feature(lakes_50m, zorder=2)
            #ax.set_extent(extent, crs=lat_lon_proj)
            if patch_count > 0: 
                neuron_subset = neuron_activations.loc[time_valid_patches, neurons_plotted]
                top_neuron = neuron_subset.idxmax(axis=1)
                for p in neuron_subset.index:
                    n_max = top_neuron[p]
                    px, py = bmap(meta_data["lon"].sel(p=p).values, meta_data["lat"].sel(p=p).values)
                    ax.pcolormesh(px, py,
                                    np.ma.array(meta_data["masks"].sel(p=p) * neuron_subset.loc[p, n_max],
                                                mask=meta_data["masks"].sel(p=p) == 0),
                                    vmin=plot_settings[n_max]["vmin"],
                                    vmax=plot_settings[n_max]["vmax"],
                                    cmap=plot_settings[n_max]["cmap"],
                                    zorder=3)
            plot_title_time(ax, title_str, run_date.to_pydatetime(), f_hour, f_hour)
            tx, ty = ax.transAxes.transform((0,0))
            fig.figimage(plt.imread('ncar.png'), xo=tx, yo=(ty-44), zorder=1000)
            plt.text(tx+10, ty-54, 'ensemble.ucar.edu', fontdict={'size':9, 'color':'#505050'}, transform=None)
            plt.savefig(join(out_full_path, f"cnn_storm_mode_f{f_hour:03d}_{region}.png"),
                        dpi=dpi, bbox_inches="tight", transparent=transparent)
            plt.close()


    return

def plot_title_time(ax, title, initdate, shr, ehr):
    fontdict = {'family':'monospace', 'size':12, 'weight':'bold'}

    # place title and times above corners of map
    x0, y1 = ax.transAxes.transform((0,1))
    x0, y0 = ax.transAxes.transform((0,0))
    x1, y1 = ax.transAxes.transform((1,1))
    ax.text(x0, y1+10, title, fontdict=fontdict, transform=None)

    initstr  = initdate.strftime('Init: %a %Y-%m-%d %H UTC') 
    if ((ehr - shr) == 0): 
        validstr = (initdate+timedelta(hours=shr)).strftime('Valid: %a %Y-%m-%d %H UTC')
    else:
        validstr1 = (initdate+timedelta(hours=(shr-1))).strftime('%a %Y-%m-%d %H UTC')
        validstr2 = (initdate+timedelta(hours=ehr)).strftime('%a %Y-%m-%d %H UTC')
        validstr = "Valid: %s - %s"%(validstr1, validstr2)

    ax.text(x1, y1+20, initstr, horizontalalignment='right', transform=None)
    ax.text(x1, y1+5, validstr, horizontalalignment='right', transform=None)
    return

