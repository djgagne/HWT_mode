"""

Copied from WRF_SPC.py Sep 20, 2019.

Given a model initialization time and a valid time, plot crefuh around hagelslag objects. 

"""
import argparse
import datetime
import pdb
import os
import sys
import pandas as pd
import numpy as np
import fieldinfo # levels and color tables - Adapted from /glade/u/home/wrfrt/wwe/python_scripts/fieldinfo.py 20190125.
from wrf import to_np, getvar, get_cartopy, latlon_coords
from metpy.units import units
from netCDF4 import Dataset
import cartopy
import matplotlib
matplotlib.use("Agg") # allows dav slurm jobs
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def counties():
    # Read my own county shape file.
    reader = cartopy.io.shapereader.Reader('/glade/work/ahijevyc/share/shapeFiles/cb_2013_us_county_500k/cb_2013_us_county_500k.shp')
    counties = list(reader.geometries())
    # Create custom cartopy feature that can be added to the axes.
    return cartopy.feature.ShapelyFeature(counties, cartopy.crs.PlateCarree())

def update_scale_labels(scale_xy, ax):
    ax = plt.gca()
    # Update labels on axes with the distance along each axis.
    # Cartopy axes do not have a set_xlabel() or set_ylabel() method. Add labels manually.
    xspan = ax.get_xlim()
    yspan = ax.get_ylim()
    xlabel = "%dkm" % (round((xspan[1]-xspan[0])/1000.))
    ylabel = "%dkm" % (round((yspan[1]-yspan[0])/1000.))
    x, y = scale_xy
    x.set_text(xlabel)
    y.set_text(ylabel)



def main():
    # =============Arguments===================
    parser = argparse.ArgumentParser(description = "Plot WRF and SPC storm reports",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--fill", type=str, default= 'crefuh', help='netCDF variable name for contour fill field')
    parser.add_argument("-b", "--barb", choices=["shr06", "wind10m",""], type=str, default="wind10m", help='wind barbs')
    parser.add_argument("-c", "--contour", type=str, default=None, help='contour field')
    parser.add_argument("-o", "--outdir", type=str, default='.', help="name of output path")
    parser.add_argument("-p", "--padding", type=float, nargs=4, help="padding on west, east, south and north side in km", 
            default=[175.,175.,175.,175.]) 
    parser.add_argument("--timeshift", type=int, default=0, help="hours to shift background field") 
    parser.add_argument("--arrow", action='store_true', help="Add storm motion vector from hagelslag")
    parser.add_argument("--no-fineprint", action='store_true', help="Don't write image details at bottom")
    parser.add_argument("--force_new", action='store_true', help="overwrite any old outfile, if it exists")
    parser.add_argument("--no-counties", action='store_true', help="Don't draw county borders (can be slow)")
    parser.add_argument("--no-mask", action='store_true', help="Don't draw object mask")
    parser.add_argument('-i', "--idir", type=str, default="/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts", 
            help="path to WRF output files")
    parser.add_argument('-s', "--stride", type=int, default=1, help="plot every stride points. speed things up with stride>1")
    parser.add_argument('-t', "--trackdir", type=str, default="/glade/scratch/ahijevyc/track_data_ncarstorm_3km_REFL_1KM_AGL_csv", 
            help="path to hagelslag track-step files")
    parser.add_argument("--patchdir", type=str, default="/glade/scratch/ahijevyc/track_data_ncarstorm_3km_REFL_1KM_AGL_nc", 
            help="path to hagelslag netCDF patches")
    parser.add_argument("initial_time", type=lambda d: datetime.datetime.strptime(d, '%Y%m%d%H'), 
            help="model initialization date and hour, yyyymmddhh")
    parser.add_argument("valid_time", type=lambda d: datetime.datetime.strptime(d, '%Y%m%d%H'), 
            help="model valid date and hour, yyyymmddhh")
    parser.add_argument("-d", "--debug", action='store_true')


    # Assign arguments to simply-named variables
    args = parser.parse_args()
    barb         = args.barb
    contour      = args.contour
    fill         = args.fill
    odir         = args.outdir
    padding      = args.padding
    timeshift    = args.timeshift
    arrow        = args.arrow
    no_fineprint = args.no_fineprint
    force_new    = args.force_new
    no_counties  = args.no_counties
    no_mask      = args.no_mask
    idir         = args.idir
    stride       = args.stride
    patchdir     = args.patchdir
    trackdir     = args.trackdir
    initial_time = args.initial_time
    valid_time   = args.valid_time
    debug        = args.debug

    if debug:
        print(args)

    # Derive lead time and make sure it is between 12 and 36 hours. 
    lead_time = valid_time - initial_time

    if lead_time < datetime.timedelta(hours=7) or lead_time > datetime.timedelta(hours=36):
        print("lead_time:",lead_time, "not between 7 and 36 hours")
        #sys.exit(1)

    # Read hagelslag track_step csv file into pandas DataFrame.
    mysterious_suffix = '' # '_13' or '_12'
    tracks = trackdir + '/' + initial_time.strftime('track_step_NCARSTORM_d01_%Y%m%d-%H%M')+mysterious_suffix+'.csv'
    if debug:
        print("reading csv file",tracks)
    df = pd.read_csv(tracks, parse_dates=['Run_Date', 'Valid_Date'])
    # Throw out everything except requested valid times.
    df = df[df.Valid_Date == valid_time]
    if df.empty:
        print("csv track step file", tracks, " has no objects at requested valid time",valid_time,". That is probably fine.")
        sys.exit(0)

    # Throw out weak UH objects
    good_UH = 25
    igood_UH = df['UP_HELI_MAX_max'] >= good_UH
    if 'UP_HELI_MIN_min' in df.columns:
        igood_UH = igood_UH | (df['UP_HELI_MIN_min'].abs() >= good_UH)
    print("ignoring",(~igood_UH).sum(),"object with abs(UH) <",good_UH)
    if debug:
        if 'UP_HELI_MIN_min' in df.columns:
            print(df[~igood_UH][["Step_ID","UP_HELI_MAX_max","UP_HELI_MIN_min"]])
        else:
            print(df[~igood_UH][["Step_ID","UP_HELI_MAX_max"]])
    df = df[igood_UH]
    if df.empty:
        print("csv track step file", tracks, " has no good UH objects at requested valid time",valid_time,". That is probably fine.")
        sys.exit(0)

    # List of all png files that will be created.
    pngfiles = odir + '/' + df.Step_ID + "_" + "{:+1.0f}".format(timeshift) + ".png"
    if all([os.path.isfile(p) for p in pngfiles]) and not force_new:
        # Exit if pngs all already exist and force_new option was not used. 
        print(initial_time, valid_time, "{:+1.0f}".format(timeshift) +"h",fill,"finished. Moving on.")
        sys.exit(0)

    if not no_mask:
        # Read netCDF patches
        patches = patchdir + '/' + initial_time.strftime('NCARSTORM_%Y%m%d-%H%M_d01_model_patches.nc')
        pnc = Dataset(patches,'r')
        masks = pnc.variables["masks"][:]
        mlons = pnc.variables["lon"][:]
        mlats = pnc.variables["lat"][:]
        mtrack_ids   = pnc.variables["track_id"][:]
        mtrack_steps = pnc.variables["track_step"][:]
        mask_centroid_lats = pnc.variables["centroid_lat"][:]
        mask_centroid_lons = pnc.variables["centroid_lon"][:]
        pnc.close()


    # Get color map, levels, and netCDF variable name appropriate for requested variable (from fieldinfo module).
    info = fieldinfo.nsc[fill]
    if debug:
        print("found nsc in fieldinfo.py. Using",info)
    cmap = colors.ListedColormap(info['cmap'])
    levels = info['levels']
    fill = info['fname'][0]

    # Get wrfout filename
    history_time = valid_time + datetime.timedelta(hours=timeshift)
    wrfout = idir + '/' + initial_time.strftime('%Y%m%d%H') + '/' + history_time.strftime('diags_d01_%Y-%m-%d_%H_%M_%S.nc')

    if debug: print("About to open "+wrfout)
    wrfnc = Dataset(wrfout,"r")
    if fill not in wrfnc.variables:
        print("variable "+ fill + " not found")
        print("choices:", wrfnc.variables.keys())
        sys.exit(1)

    # Get a 2D var from wrfout file. It has projection info.
    if debug:
        print("getvar...")
    cvar = getvar(wrfnc,fill)
    wrflat, wrflon = latlon_coords(cvar)
    # get cartopy mapping object
    if debug: print("get_cartopy...")
    WRF_proj = get_cartopy(cvar)
    fineprint0 = 'fill '+fill+" ("+cvar.units+") "

    if 'units' in info.keys():
        cvar.metpy.convert_units(info['units'])

    if hasattr(cvar, 'long_name'):
        label = cvar.long_name
    elif hasattr(cvar, 'description'):
        label = cvar.description


    # convert WRF lat/lons to x,y
    pts = WRF_proj.transform_points(cartopy.crs.PlateCarree(), to_np(wrflon[::stride,::stride]), to_np(wrflat[::stride,::stride])) # Transform lon/lat to x and y (in meters) in WRF projection.
    x, y, z = pts[:,:,0], pts[:,:,1], pts[:,:,2]

    fig = plt.figure(figsize=(10,10))
    if debug: print("plt.axes()")
    ax = plt.axes(projection=WRF_proj)
    ax.add_feature(cartopy.feature.STATES.with_scale('10m'), linewidth=0.35, alpha=0.55)

    # Set title (month and hour)
    ax.set_title(history_time.strftime("%b %HZ"))

    # Empty fineprint placeholder in lower left corner of image.
    fineprint_obj = plt.annotate(text=fineprint0, xy=(0,5), xycoords=('axes fraction', 'figure pixels'), va="bottom", fontsize=4)

    if cvar.min() > levels[-1] or cvar.max() < levels[0]:
        print('levels',levels,'out of range of cvar', cvar.values.min(), cvar.values.max())
        sys.exit(1)
    if debug: 
        print('levels:',levels, 'cmap:', cmap.colors)

    if debug:
        print("plotting filled contour",cvar.name,"...")
        
    cfill = ax.contourf(x, y, to_np(cvar[::stride,::stride]), levels=levels, cmap=cmap)

    # Color bar
    cb = plt.colorbar(cfill, ax=ax, format='%.0f', shrink=0.52, orientation='horizontal')
    if hasattr(cvar,"units"):
        cb.set_label(label+" ("+cvar.units+")", fontsize="small")
    if len(levels) < 10:
        # label every level if there is room.
        cb.set_ticks(levels)
    cb.ax.tick_params(labelsize='xx-small')
    cb.outline.set_linewidth(0.5)

    # Create 2 annotation object placeholders for spatial scale. Will be updated with each set_extent().
    scale_kw = {"ha":"center","rotation_mode":"anchor","xycoords":"axes fraction","textcoords":"offset points"}
    scale_xy = ( ax.annotate("", (0.5, 0), xytext=(0,-5), va='top',  rotation='horizontal', **scale_kw), 
                 ax.annotate("", (0, 0.5), xytext=(-5,0), va='bottom', rotation='vertical', **scale_kw) )


    # Special case of composite reflectivity, UH overlay
    if args.fill == 'crefuh':
        max_uh = getvar(wrfnc,info['fname'][1])
        min_uh = getvar(wrfnc,info['fname'][2])
        max_uh_threshold = info['max_threshold']
        min_uh_threshold = info['min_threshold']
        print("UH max:", max_uh.max().values)
        print("UH min:", min_uh.min().values)
        if max_uh.max() > max_uh_threshold:
            print("Filled contour UH >",max_uh_threshold)
            # Don't use contourf if the data fall outside the levels range. You will get ValueError: 'bboxes' cannot be empty.
            # See https://github.com/SciTools/cartopy/issues/1290
            cs1 = ax.contourf(x, y, to_np(max_uh), levels=[max_uh_threshold,1000], colors='black', 
                    alpha=0.3 )
            if debug: print("solid contour UH >",max_uh_threshold)
            cs2 = ax.contour(x, y, to_np(max_uh), levels=max_uh_threshold*np.arange(1,6), colors='black', 
                    linestyles='solid', linewidths=0.4 )
            fineprint0 += "UH>"+str(max_uh_threshold) +" "+ max_uh.units + " "
            # Oddly, the zero contour is plotted if there are no other valid contours
            if 0.0 in cs2.levels:
                print("uh has zero contour for some reason. Hide it")
                if debug:
                    pdb.set_trace()
                for i in cs2.collections: i.remove()

        if min_uh.min() < min_uh_threshold:
            print("Filled UH contour <",min_uh_threshold)
            # Don't use contourf if the data fall outside the levels range. You will get ValueError: 'bboxes' cannot be empty. 
            # See https://github.com/SciTools/cartopy/issues/1290
            negUH1 = ax.contourf(x, y, to_np(min_uh), levels=[-1000, min_uh_threshold], colors='black', 
                    alpha=0.3 )
            if debug: print("dashed contour UH <",min_uh_threshold)
            negUH2 = ax.contour(x, y, to_np(min_uh), levels=min_uh_threshold*np.arange(6,0,-1), colors='black', 
                    linestyles='dashed', linewidths=0.4 )
            fineprint0 += "UH<"+str(-min_uh_threshold) +" "+ min_uh.units + " " 
            if 0.0 in negUH2.levels:
                print("neg uh has a zero contour. Hide it")
                if debug:
                    pdb.set_trace()
                for i in negUH2.collections: i.remove()

    if not no_counties:
        if debug:
            print("About to draw counties")
        ax.add_feature(counties(), facecolor="none", edgecolor='black', alpha=0.25, linewidth=0.2)

    if barb:
        # Get barb netCDF variable name appropriate for requested variable (from fieldinfo module).
        info = fieldinfo.nsc[barb]
        if debug:
            print("found nsc in fieldinfo.py. Using",info)
        if args.barb == 'wind10m': u,v = getvar(wrfnc, 'uvmet10', units='kt')
        if args.barb == 'shr06':
            u = getvar(wrfnc, 'USHR6')*1.93
            v = getvar(wrfnc, 'VSHR6')*1.93
            u.attrs['units'] = 'kt'
            v.attrs['units'] = 'kt'

        # Density of barbs stays the same, no matter the domain size (padding)
        # larger domain = greater stride
        skip = int(round(np.max([(padding[0]+padding[1]), (padding[2]+padding[3])])/50))

        if args.fill ==  'crefuh': alpha=0.6
        else: alpha=1.0

        if debug: print("plotBarbs: starting barbs")
        # barbs already oriented with map projection. In Basemap, we needed to use m.rotate_vector().
        cs2 = ax.barbs(x[::skip*stride,::skip*stride], y[::skip*stride,::skip*stride], 
                to_np(u)[::skip*stride,::skip*stride], to_np(v)[::skip*stride,::skip*stride], color='black', 
                alpha=alpha, length=5, linewidth=0.25, sizes={'emptybarb':0.05} )
        fineprint0 += "wind barb (" + u.units + ") "

    if contour:
        # Get netCDF variable name appropriate for requested variable from fieldinfo module.
        info = fieldinfo.nsc[contour]
        if debug:
            print("found nsc in fieldinfo.py. Using",info)
        cvar = getvar(wrfnc, info['fname'][0])
        if 'units' in info.keys():
            cvar.metpy.convert_units(info['units'])
        levels = info['levels']
        # could use levels from fieldinfo module, but default is often less cluttered.
        alpha=0.4

        if debug: print("starting "+contour+" contours")
        cr = ax.contour(x[::stride,::stride], y[::stride,::stride], 
                cvar[::stride,::stride], levels=levels, colors='black', alpha=alpha, 
                linewidths=0.75)
        clab = ax.clabel(cr, inline=False, fmt='%.0f', fontsize=6)
        fineprint0 += "contour "+contour+" (" + cvar.units + ") "

    for lon,lat,stepid,trackid,u,v,pngfile in zip(df.Centroid_Lon, df.Centroid_Lat,df.Step_ID,df.Track_ID,df.Storm_Motion_U,df.Storm_Motion_V,pngfiles):

        fineprint = fineprint0 + "\nwrfout " + os.path.realpath(wrfout)
        if not no_mask:
            fineprint += "\npatches "+patches
        fineprint += "\ntracks "+tracks
        fineprint += "\ntrackid "+trackid
        fineprint += "\ncreated "+str(datetime.datetime.now(tz=None)).split('.')[0]

        if not no_fineprint: # show fineprint
            fineprint_obj.set_text(fineprint)

        x, y = WRF_proj.transform_point(lon, lat, cartopy.crs.PlateCarree()) # Transform lon/lat to x and y (in meters) in WRF projection.
        ax.set_extent([x-padding[0]*1000., x+padding[1]*1000., y-padding[2]*1000., y+padding[3]*1000.], crs=WRF_proj)

        track_id_int = int(trackid.split('_')[-1])
        step_id_int = int(stepid.split('_')[-1])

        # Contour object mask
        if not no_mask:
            # Find matching mask track id and step. For some reason, steps start with 1 in netCDF patches file
            matches = (mtrack_ids == track_id_int) & (mtrack_steps == step_id_int+1)
            ip = np.where(matches)[0][0]
            if not any(matches):
                pdb.set_trace()
            tolerance = 0.025 # TODO: figure out why centroid of csv object and nc patch differ at all
            if np.abs(lon-mask_centroid_lons[ip]) > tolerance:
                print(stepid,lon,mask_centroid_lons[ip])
            if np.abs(lat-mask_centroid_lats[ip]) > tolerance:
                print(stepid,lat,mask_centroid_lats[ip])
            mask = masks[ip]
            mlon = mlons[ip]
            mlat = mlats[ip]
            mcntr = ax.contour(mlon, mlat, mask, levels=[0,10], colors='black', alpha=0.6, 
                    linewidths=2., linestyles="solid", zorder=2, transform=cartopy.crs.PlateCarree())

        # Update axes labels (distance along axes).
        update_scale_labels(scale_xy, ax)

        if arrow:
            # Storm motion vector points from previous location to present location.
            smv = ax.arrow(x-u, y-v, u, v, color=mcntr.colors, alpha=mcntr.get_alpha(), # Can't get head to show. Tried quiver, plot, head_width, head_length..., annotate... 
                 linewidth=1, zorder=2, capstyle='round', transform=WRF_proj) # tried length_includes_head=True, but zero-size gives ValueError about shape Nx2 needed.


        # Save image. 
        plt.savefig(pngfile, dpi=175)
        print('created ' + os.path.realpath(pngfile))

        if arrow: smv.remove()

        # Remove object mask contour
        if not no_mask:
            for i in mcntr.collections: i.remove()


    if debug: pdb.set_trace()
    plt.close(fig)
    print("to sort -2 -1 +0 +1 +2 numerically:")
    print("ls d01*png | sort -g -k 1."+str(len(stepid)+2))
    print("to trim whitespace:")
    print("convert -crop 980x1012+390+173 in.png out.png")

if __name__ == "__main__":
    main()
