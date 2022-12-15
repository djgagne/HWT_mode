import numpy as np
import tensorflow as tf
import xarray as xr
import pandas as pd
import datetime as dt
#from mpl_toolkits.basemap import Basemap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import sys, pickle, time, math, os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import multiprocessing
from netCDF4 import Dataset
from pathlib import Path
import pygrib, json

import warnings
warnings.filterwarnings("ignore")

#ax.set_extent([-120.81058,-75.704895,23.159264,48.753662]) #corners of wrf grid
domains = { 'CONUS': [-120.81058,-75.4,23.159264,49.4],
            'SGP' : [-109,-86,25.5,38],
            'NGP' : [-108,-86,40,49.80],
            'CGP' : [-108.50,-86.60,33,45],
            'NW'  : [-124,-102.1,39,49],
            'SW'  : [-122,-101,31,42],
            'SE'  : [-96,-77,26.75,37],
            'NE'  : [-92,-69,39,47.5],
            'MATL': [-92,-72,33.5,41.50] }

def readcm(name):
    '''Read colormap from file formatted as 0-1 RGB CSV'''
    rgb = []
    fh = open(name, 'r')
    for line in fh.read().splitlines(): rgb.append(list(map(float,line.split())))
    return rgb

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    rgb, appending = [], False
    fh = open(os.getenv('NCARG_ROOT','/glade/u/apps/dav/opt/ncl/6.6.2/intel/19.0.5')+'/lib/ncarg/colormaps/%s.rgb'%name, 'r') # CSS made variable, commented out previous line
    for line in fh.read().splitlines():
        if appending: rgb.append(list(map(float,line.split())))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def log(msg): print(time.ctime(time.time()), msg)

def get_hrrr_data(this_fhr):
    fname = '/glade/scratch/sobash/HRRR/%s/hrrr.t%sz.wrfsfcf%02d.grib2'%(yyyymmddhh,hh,this_fhr)
    fh = pygrib.open(fname)
    cref = fh[1].values
    hmuh = fh[45].values
    if this_fhr > 1:
        ushr = fh[140].values
        vshr = fh[141].values
        mucape = fh[157].values
    else:
        ushr = fh[137].values
        vshr = fh[138].values
        mucape = fh[154].values
    fh.close()

    ushr, vshr = ushr*1.94, vshr*1.94
    return (cref, hmuh, mucape, ushr, vshr)

def make_mode_dotplots(this_fhr):
    log('reading %s %s'%(yyyymmddhh,this_fhr))

    # create blank figure
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([x,y,w,h], projection=axes_proj)
    for i in ax.spines.items(): i[1].set_linewidth(0.5)

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.1)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), edgecolor='k', linewidth=0.25, facecolor='k', alpha=0.1)

    # create time lagged forecast times
    time_lagged_members = []
    for i,new_fhr in enumerate(range(this_fhr,this_fhr+49)):
        new_init = thisdate - dt.timedelta(hours=i)
        fname = '/glade/scratch/cbecker/HRRR_HWT_2022_real_time/model_output/labels/model_labels_%s.parquet'%new_init.strftime('%Y-%m-%d_%H00')
        #fname = '/glade/work/cbecker/HWT/HWT_evaluation_HRRR/HWT_May2019_hourly/labels/model_labels_%s.parquet'%new_init.strftime('%Y-%m-%d_%H00')
        if os.path.exists(fname):
            df = pd.read_parquet(fname)
            df = df[df['Forecast_Hour']==new_fhr]
            if len(df.index) > 0: time_lagged_members.append( df )
    print('time lagged ensemble has %d members'%len(time_lagged_members))

    print(df)
    # concatenate time-lagged members and compute mean
    df = pd.concat(time_lagged_members, ignore_index=True)
    
    models = ['CNN', 'DNN', 'SS']
    dotcolors = {'Supercell':plt.get_cmap('Reds'), 'QLCS':plt.get_cmap('Blues'), 'Disorganized':plt.get_cmap('Greens')}
    levels = np.arange(0,1.1,0.1)
    norm = colors.BoundaryNorm(levels, plt.get_cmap('Reds').N)

    for model in models:
        if model == 'CNN': model_fname = 'CNN_1'
        elif model == 'DNN': model_fname = 'DNN_1'
        elif model == 'SS': model_fname = 'SS'

        all_dots = []
        for mode in ['Supercell', 'QLCS', 'Disorganized']:
            df_mask = ( df['%s_label'%model_fname] == mode )
            if len(df_mask.index) > 0:
                this_df = df[df_mask]
                cs_dots = ax.scatter(this_df['Centroid_Lon'], this_df['Centroid_Lat'], c=this_df['%s_%s_prob'%(model_fname,mode)].values, s=35, \
                                     cmap=dotcolors[mode], norm=norm, transform=ccrs.PlateCarree(), alpha=0.7)
            all_dots.append(cs_dots)

        # loop through domains and save figure 
        for dom in domains.keys():
            ax.set_extent(domains[dom])
            ax_x, ax_y, ax_w, ax_h = ax.get_position().bounds #get position of axes within figure
            fontdict = {'family':'Helvetica', 'size':14, 'fontweight':'bold'}
            fontdict2 = {'family':'Helvetica', 'size':12}

            # add title and init/valid text
            t1 = fig.text(ax_x, ax_y-0.025, 'HRRR-TL %s Convective Mode Centroids'%model, fontdict=fontdict, transform=fig.transFigure)

            initstr  = thisdate.strftime('Init: %a %Y-%m-%d %H UTC')
            validstr = (thisdate+dt.timedelta(hours=this_fhr)).strftime('Valid: %a %Y-%m-%d %H UTC')
            t2 = fig.text(ax_x, ax_y-0.045, initstr, fontdict=fontdict2, transform=fig.transFigure)
            t3 = fig.text(ax_x, ax_y-0.065, validstr, fontdict=fontdict2, transform=fig.transFigure)

            t4 = fig.text(ax_x+0.01, ax_y+0.01, 'Number of Members = %d'%len(time_lagged_members), fontdict={'family':'Helvetica', 'size':12}, transform=fig.transFigure)

            # add colorbar 
            cax = fig.add_axes([ax_x+0.6*ax_w,ax_y-0.03,ax_w/3.0,0.02])
            #if 'mode-objects' in plot:
            #    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=props['cmap']), cax=cax, orientation='horizontal', ticks=props['levels'][:-1], extendfrac=0.0)
            #else:

            cb = plt.colorbar(cs_dots, cax=cax, orientation='horizontal', ticks=levels, extendfrac=0.0)
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(labelsize=9)

            plt.savefig("%s/%s/%s_modedots_f%03d_%s.png"%(graphics_dir,yyyymmddhh,model,this_fhr, dom), dpi=dpi, bbox_inches='tight')

            for p in all_dots: p.remove()
            for t in [t1, t2, t3, t4]: t.remove()
            cax.remove()

    plt.close()

def make_plot_probs(this_fhr):
    log('reading %s %s'%(yyyymmddhh,this_fhr))

    # create blank figure
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([x,y,w,h], projection=axes_proj)
    for i in ax.spines.items(): i[1].set_linewidth(0.5)

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.1)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), edgecolor='k', linewidth=0.25, facecolor='k', alpha=0.1) 

    # create time lagged forecast times
    time_lagged_members = []
    for i,new_fhr in enumerate(range(this_fhr,this_fhr+49)):
    #for i,new_fhr in enumerate(range(this_fhr,this_fhr+1)):
        new_init = thisdate - dt.timedelta(hours=i)
        #fname = '/glade/work/cbecker/HWT/HWT_evaluation_HRRR/HWT_May2019_hourly/evaluation/%s00/label_probabilities_%s00_fh_%02d.nc'%(new_init.strftime('%Y%m%d%H'),new_init.strftime('%Y%m%d%H'),new_fhr)
        fname = '/glade/scratch/cbecker/HRRR_HWT_2022_real_time/model_output/evaluation/%s00/label_probabilities_%s00_fh_%02d.nc'%(new_init.strftime('%Y%m%d%H'),new_init.strftime('%Y%m%d%H'),new_fhr)
        if os.path.exists(fname): time_lagged_members.append( xr.open_dataset(fname) )

    print('time lagged ensemble has %d members'%len(time_lagged_members))

    # concatenate time-lagged members
    combined = xr.concat(time_lagged_members, dim='time')

    # compute time-lagged mean
    combined = combined.mean(dim='time')

    gridded_mode_data = xr.load_dataset('/glade/scratch/cbecker/HRRR_HWT_2022_real_time/model_output/evaluation/%s00/label_probabilities_%s00_fh_01.nc'%(yyyymmddhh,yyyymmddhh))
    lats, lons = gridded_mode_data['lat'], gridded_mode_data['lon']
     
    models = ['ML1', 'ML2', 'ML3']

    for model in models:
        if model == 'ML1': model_fname = 'CNN_1'
        elif model == 'ML2': model_fname = 'DNN_1'
        elif model == 'ML3': model_fname = 'SS'
       
        if model == 'MEAN':
            nprobs_supercell = combined['CNN_1_Supercell_nprob'].values + combined['DNN_1_Supercell_nprob'].values + combined['SS_Supercell_nprob'].values
            nprobs_qlcs = combined['CNN_1_QLCS_nprob'].values + combined['DNN_1_QLCS_nprob'].values + combined['SS_QLCS_nprob'].values
            nprobs_disorganized = combined['CNN_1_Disorganized_nprob'].values + combined['DNN_1_Disorganized_nprob'].values + combined['SS_Disorganized_nprob'].values
            nprobs_supercell, nprobs_qlcs, nprobs_disorganized = nprobs_supercell/3.0, nprobs_qlcs/3.0, nprobs_disorganized/3.0
        else:
            nprobs_supercell = combined['%s_Supercell_nprob'%model_fname].values
            nprobs_qlcs = combined['%s_QLCS_nprob'%model_fname].values
            nprobs_disorganized = combined['%s_Disorganized_nprob'%model_fname].values

        props = {'levels': [0.02,0.05,0.1,0.2,0.3,0.4,0.5],\
                 'cmap': colors.ListedColormap( readcm('/glade/u/home/sobash/hwt2020_graphics/cmap_rad.rgb')[1:14] ) }
        norm = colors.BoundaryNorm(props['levels'], props['cmap'].N)
    
        all_paths = []
        if nprobs_supercell.max() >0.02:
            cs_paths = ax.contour(lons, lats, nprobs_supercell, levels=props['levels'], colors='red', linewidths=3, transform=ccrs.PlateCarree(), alpha=0.9)
            all_paths.append(cs_paths)
        if nprobs_disorganized.max() >0.02:
            cs_paths2 = ax.contour(lons, lats, nprobs_disorganized, levels=props['levels'], colors='green', linewidths=3, transform=ccrs.PlateCarree(), alpha=0.9)
            all_paths.append(cs_paths2)
        if nprobs_qlcs.max() >0.02:
            cs_paths3 = ax.contour(lons, lats, nprobs_qlcs, levels=props['levels'], colors='blue', linewidths=3, transform=ccrs.PlateCarree(), alpha=0.9)
            all_paths.append(cs_paths3)
              
        # loop through domains and save figure 
        for dom in domains.keys():
            ax.set_extent(domains[dom])
            ax_x, ax_y, ax_w, ax_h = ax.get_position().bounds #get position of axes within figure
            fontdict = {'family':'Helvetica', 'size':14, 'fontweight':'bold'}
            fontdict2 = {'family':'Helvetica', 'size':12}

            # add title and init/valid text
            t1 = fig.text(ax_x, ax_y-0.025, 'HRRR-TL %s Convective Mode Smoothed Neighborhood Probability'%model, fontdict=fontdict, transform=fig.transFigure)
    
            initstr  = thisdate.strftime('Init: %a %Y-%m-%d %H UTC')
            validstr = (thisdate+dt.timedelta(hours=this_fhr)).strftime('Valid: %a %Y-%m-%d %H UTC')
            t2 = fig.text(ax_x, ax_y-0.045, initstr, fontdict=fontdict2, transform=fig.transFigure)
            t3 = fig.text(ax_x, ax_y-0.065, validstr, fontdict=fontdict2, transform=fig.transFigure) 
            
            t4 = fig.text(ax_x+0.01, ax_y+0.01, 'Number of Members = %d'%len(time_lagged_members), fontdict={'family':'Helvetica', 'size':12}, transform=fig.transFigure)
            
            # add colorbar 
            cax = fig.add_axes([ax_x+0.6*ax_w,ax_y-0.03,ax_w/3.0,0.02])
            #if 'mode-objects' in plot:
            #    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=props['cmap']), cax=cax, orientation='horizontal', ticks=props['levels'][:-1], extendfrac=0.0)
            #else:
                
            cb = plt.colorbar(all_paths[-1], cax=cax, orientation='horizontal', ticks=props['levels'], extendfrac=0.0)
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(labelsize=9)
            
            plt.savefig("%s/%s/HRRR_mode-nprob-%s_f%03d_%s.png"%(graphics_dir,yyyymmddhh,model.lower(),this_fhr, dom), dpi=dpi, bbox_inches='tight')
          
            for p in all_paths:
                if hasattr(p, 'collections'):
                    for tp in p.collections: tp.remove() 
            for t in [t1, t2, t3, t4]: t.remove()
            cax.remove()
 
    plt.close()

# create summary plot for Day 1 forecast
def make_plot_summary():
    base_url = '/glade/work/cbecker/HWT/HWT_evaluation_HRRR/HWT_May2019_hourly/evaluation/'

    all_inits = []    
    for i,new_fhr in enumerate(range(0,48)):
        new_init = thisdate - dt.timedelta(hours=i)
        print('Reading', new_init)

        if new_init.hour in [0,6,12,18]: fhours=range(1,49) 
        else: fhours = range(1,19)

        this_init_hours = []
        for fhr in fhours: 
            fname = '%s/%s00/label_probabilities_%s00_fh_%02d.nc'%(base_url,new_init.strftime('%Y%m%d%H'),new_init.strftime('%Y%m%d%H'),fhr)
            if os.path.exists(fname): this_init_hours.append( xr.open_dataset(fname) )
            
        this_init_ds = xr.concat(this_init_hours, dim='time')

        #this_init_ds = this_init_ds.where(this_init_ds['valid_times'] >= np.datetime64(stime), drop=True)
        #this_init_ds = this_init_ds.where(this_init_ds['valid_times'] < np.datetime64(etime), drop=True)

        print(this_init_ds)

        all_inits.append(this_init_ds)

def make_plot(this_fhr):
    log('reading %s %s'%(yyyymmddhh,this_fhr))

    # create blank figure
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([x,y,w,h], projection=axes_proj)
    for i in ax.spines.items(): i[1].set_linewidth(0.5)

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.1)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), edgecolor='k', linewidth=0.25, facecolor='k', alpha=0.1)

    # get data
    cref, hmuh, mucape, ushr, vshr = get_hrrr_data(this_fhr)

    # read in CSV files with object predictions and json files with coordinates for object boundaries and match
    fname = '/glade/scratch/cbecker/HRRR_HWT_2022_real_time/model_output/labels/model_labels_%s.csv'%(thisdate.strftime('%Y-%m-%d_%H00'))
    df = pd.read_csv(fname)
    df = df[df['Forecast_Hour'] == this_fhr]
    df['coordinates'] = 0

    fname = '/glade/scratch/cbecker/HRRR_HWT_2022_real_time/track_data_hrrr_3km_json_refl/HRRR-ZARR_oper_%s.json'%(thisdate.strftime('%Y%m%d-%H00'))
    data = json.load(open(fname, 'r'))
    for d in data['features']:
        lat, lon, fhr = d['properties']['centroid_lat'], d['properties']['centroid_lon'], d['properties']['valid_time']
        if fhr == this_fhr:
            df_mask = ( df['Centroid_Lon'] == round(lon, 5) ) & ( df['Centroid_Lat'] == round(lat, 5) )
            df.loc[df_mask,'coordinates'] = d['geometry']['coordinates']

    # plot contour field
    cs_paths, cs_lines, cs_dots, cs_barbs = [], [], [], []
    plot_types = ['crefuh', 'hmuh', 'capeshr', 'mode-objects-ml1-supercell', 'mode-objects-ml2-supercell', 'mode-objects-ml3-supercell', \
                                               'mode-objects-ml1-qlcs', 'mode-objects-ml2-qlcs', 'mode-objects-ml3-qlcs', \
                                               'mode-objects-ml1-disorg', 'mode-objects-ml2-disorg', 'mode-objects-ml3-disorg']

    for plot in plot_types:
        if plot == 'crefuh':
            title = 'Composite reflectivity and UH > 25'
            props = {'levels': [5,10,15,20,25,30,35,40,45,50,55,60,65,70],\
                   'cmap': colors.ListedColormap( readcm('/glade/u/home/sobash/hwt2020_graphics/cmap_rad.rgb')[1:14] ) }
            norm = colors.BoundaryNorm(props['levels'], props['cmap'].N)
            cs_paths = ax.contourf(pts_x, pts_y, cref, levels=props['levels'], cmap=props['cmap'], norm=norm, extend='max')
            if hmuh.max() > 25: cs_lines = ax.contour(pts_x, pts_y, hmuh, colors='k', levels=[25,100,1000])

        elif plot == 'hmuh':
            title = 'Hourly-max updraft helicity'
            props = {'levels': [10,25,50,75,100,125,150,175,200,250,300,400,500],
                   'cmap': colors.ListedColormap( readNCLcm('prcp_1')[1:15] ) }
            norm = colors.BoundaryNorm(props['levels'], props['cmap'].N)

            cs_paths = ax.contourf(pts_x, pts_y, hmuh, levels=props['levels'], cmap=props['cmap'], norm=norm, extend='max')

        elif plot == 'capeshr':
            title = 'MUCAPE and 0-6km shear vector'
            props = {'levels': [100,250,500,750,1000,1250,1500,1750,2000,2500,3000,3500,4000,4500,5000,5500,6000],
                   'cmap': colors.ListedColormap( ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1] ) }
            norm = colors.BoundaryNorm(props['levels'], props['cmap'].N)
            skip = 40
            cs_paths = ax.contourf(pts_x, pts_y, mucape, levels=props['levels'], cmap=props['cmap'], norm=norm, extend='max')
            cs_barbs = ax.barbs(pts_x[::skip,::skip], pts_y[::skip,::skip], ushr[::skip,::skip], vshr[::skip,::skip], color='k', alpha=0.7, length=5.5, linewidth=0.25, sizes={'emptybarb':0.05})

        elif 'mode-object' in plot:
            mode_types = { 'qlcs':0, 'supercell':1, 'disorg':2 }
            mod = plot.split("-")
            ml_model = mod[2]
            this_mode = mod[3]

            if ml_model == 'ml1': predictions = df[['CNN_1_QLCS_prob', 'CNN_1_Supercell_prob', 'CNN_1_Disorganized_prob']].values[:,mode_types[this_mode]]
            elif ml_model == 'ml2': predictions = df[['DNN_1_QLCS_prob', 'DNN_1_Supercell_prob', 'DNN_1_Disorganized_prob']].values[:,mode_types[this_mode]]
            elif ml_model == 'ml3': predictions = df[['SS_QLCS_prob', 'SS_Supercell_prob', 'SS_Disorganized_prob']].values[:,mode_types[this_mode]]
            df['predictions'] = predictions

            props = {'levels': [5,10,15,20,25,30,35,40,45,50,55,60,65,70],\
                   'cmap': colors.ListedColormap( readcm('/glade/u/home/sobash/hwt2020_graphics/cmap_rad.rgb')[1:14] ) }
            norm = colors.BoundaryNorm(props['levels'], props['cmap'].N)

            cs_paths = ax.contourf(pts_x, pts_y, cref, levels=props['levels'], cmap=props['cmap'], norm=norm, extend='max', alpha=0.1)

            props   = { 'levels':np.arange(0.1,1.01,0.1), 'cmap':plt.get_cmap('Blues') }
            if this_mode == 'supercell':
                title = 'Mode Probabilities -- %s / Supercell'%ml_model.upper()
                props['cmap'] = plt.get_cmap('Reds')
            elif this_mode == 'qlcs':
                title = 'Mode Probabilities -- %s / QLCS'%ml_model.upper()
                props['cmap'] = plt.get_cmap('Blues')
            elif this_mode == 'disorg':
                title = 'Mode Probabilities -- %s / Disorganized'%ml_model.upper()
                props['cmap'] = plt.get_cmap('Greens')
            norm  = colors.BoundaryNorm(props['levels'], props['cmap'].N)

            # plot patches color coded by probability
            for ind, row in df.iterrows():
                this_color = props['cmap'](norm(row['predictions']))
                these_lons, these_lats = np.array(row['coordinates'])[:,0], np.array(row['coordinates'])[:,1]
                cs_lines = ax.fill(these_lons, these_lats, color=this_color, edgecolor='0.8', linewidth=0.75, transform=ccrs.PlateCarree())
                #cs_dots = ax.pcolormesh(lon[storm_idx,:][0,:], lat[storm_idx,:][0,:], masked_patch, norm=norm, cmap=props['cmap'], zorder=3, shading='nearest', transform=ccrs.PlateCarree())

        # loop through domains and save figure 
        for dom in domains.keys():
            ax.set_extent(domains[dom])
            ax_x, ax_y, ax_w, ax_h = ax.get_position().bounds #get position of axes within figure
            fontdict = {'family':'Helvetica', 'size':14, 'fontweight':'bold'}
            fontdict2 = {'family':'Helvetica', 'size':10}

            # add title and init/valid text
            t1 = fig.text(ax_x, ax_y-0.025, 'HRRR '+title, fontdict=fontdict, transform=fig.transFigure)

            initstr  = thisdate.strftime('Init: %a %Y-%m-%d %H UTC')
            validstr = (thisdate+dt.timedelta(hours=this_fhr)).strftime('Valid: %a %Y-%m-%d %H UTC')
            t2 = fig.text(ax_x, ax_y-0.045, initstr, fontdict=fontdict2, transform=fig.transFigure)
            t3 = fig.text(ax_x, ax_y-0.065, validstr, fontdict=fontdict2, transform=fig.transFigure)

            # add colorbar 
            cax = fig.add_axes([ax_x+ax_w/2.0,ax_y-0.03,ax_w/2.0,0.02])
            if 'mode-objects' in plot:
                cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=props['cmap']), cax=cax, orientation='horizontal', ticks=props['levels'][:-1], extendfrac=0.0)
            else:
                cb = plt.colorbar(cs_paths, cax=cax, orientation='horizontal', ticks=props['levels'][:-1], extendfrac=0.0)
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(labelsize=9)

            out_name = "%s/%s/HRRR_%s_f%03d_%s.png"%(graphics_dir,yyyymmddhh,plot,this_fhr,dom)
            plt.savefig(out_name, dpi=dpi, bbox_inches='tight')
            #log('saved %s'%out_name)

            for t in [t1, t2, t3]: t.remove()
            cb.remove()

       # remove what we just plotted
        if hasattr(cs_paths, 'collections'):
            for tp in cs_paths.collections: tp.remove()
        if hasattr(cs_lines, 'collections'):
            for tp in cs_lines.collections: tp.remove()
        if hasattr(cs_dots, 'collections'):
            for tp in cs_dots.collections: tp.remove()
        if plot == 'capeshr':
            cs_barbs.remove()

        cs_paths, cs_lines, cs_dots, cs_barbs = [], [], [], []

    plt.close()


thisdate = dt.datetime.strptime(sys.argv[1], '%Y%m%d%H')
yyyymmddhh = thisdate.strftime('%Y%m%d%H')
yyyymmdd = thisdate.strftime('%Y%m%d')
hh = thisdate.strftime('%H')
#model = sys.argv[2]
gtype = sys.argv[2]
graphics_dir = "/glade/scratch/sobash/realtime_graphics"
Path("%s/%s"%(graphics_dir,yyyymmddhh)).mkdir(parents=True, exist_ok=True)

aspect = 0.6231716825706464 #only for US
#dpi, fig_width_pixels = 90, 1080
dpi, fig_width_pixels = 90, 1200
fig_width  = fig_width_pixels/float(dpi)
fig_height = fig_width * aspect + 0.93
x,y,w,h = 0.01, 0.7/float(fig_height), 0.98, 0.98 * fig_width * aspect/float(fig_height) 

axes_proj = ccrs.LambertConformal(central_longitude=-101,central_latitude=38.33643,standard_parallels=(32,46))

with np.load('hrrr_latlons.npz') as data:
    lats, lons, pts_x, pts_y = data['lats'], data['lons'], data['pts_x'], data['pts_y']

# use multiprocess to run different forecast hours in parallel
print('running graphics in parallel')
fhr_list = range(1,25)
nprocs    = 8
chunksize = int(math.ceil(len(fhr_list) / float(nprocs)))

# running pool twice would occasionally hang...
if gtype == 'obprobs':
    pool      = multiprocessing.Pool(processes=nprocs)
    data      = pool.map(make_plot, fhr_list, chunksize)
    pool.close()
 
if gtype == 'nprobs':
    # only make these plots for CONUS domain
    domains = { 'CONUS': [-120.81058,-75.4,23.159264,49.4] }
    pool      = multiprocessing.Pool(processes=nprocs)
    data      = pool.map(make_plot_probs, fhr_list, chunksize)
    pool.close()
