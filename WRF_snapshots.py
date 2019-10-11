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
from netCDF4 import Dataset
import cartopy
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# =============Arguments===================
parser = argparse.ArgumentParser(description = "Plot WRF and SPC storm reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--fill", type=str, default= 'crefuh', help='netCDF variable name for contour fill field')
parser.add_argument("-b", "--barb", choices=["wind10m"], type=str, help='wind barbs')
parser.add_argument("-o", "--outdir", type=str, default='.', help="name of output path")
parser.add_argument("-p", "--padding", type=float, nargs=4, help="padding on west, east, south and north side in km", default=[100.,100.,100.,100.]) 
parser.add_argument("--force_new", action='store_true', help="overwrite any old outfile, if it exists")
parser.add_argument("--counties", action='store_true', help="draw county borders (can be slow)")
parser.add_argument('-i', "--idir", type=str, default="/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts", help="path to WRF output files")
parser.add_argument('-t', "--tdir", type=str, default="/glade/work/sobash/NSC_objects/track_data_ncarstorm_3km_csv_refl", help="path to hagelslag track-step files")
parser.add_argument("initial_time", type=lambda d: datetime.datetime.strptime(d, '%Y%m%d%H'), help="model initialization date and hour, yyyymmddhh")
parser.add_argument("valid_time", type=lambda d: datetime.datetime.strptime(d, '%Y%m%d%H'), help="model valid date and hour, yyyymmddhh")
parser.add_argument("-d", "--debug", action='store_true')


# Assign arguments to simple-named variables
args = parser.parse_args()
fill = args.fill
barb = args.barb
odir = args.outdir
padding = args.padding
force_new = args.force_new
counties = args.counties
idir = args.idir
tdir = args.tdir
initial_time = args.initial_time
valid_time = args.valid_time
debug = args.debug

if debug:
    print(args)

# Derive lead time and make sure it is between 12 and 36 hours. 
lead_time = valid_time - initial_time

if lead_time < datetime.timedelta(hours=12) or lead_time > datetime.timedelta(hours=36):
    print("lead_time:",lead_time, "not between 12 and 36 hours")
    sys.exit(1)

def change_scale(scale_xy):
    # Update labels on axes with the distance along each axis.
    # Cartopy axes do not have a set_xlabel() or set_ylabel() method. Add labels manually.
    xspan = ax.get_xlim()
    yspan = ax.get_ylim()
    xlabel = "%dkm" % (np.round((xspan[1]-xspan[0])/1000.))
    ylabel = "%dkm" % (np.round((yspan[1]-yspan[0])/1000.))
    x, y = scale_xy
    x.set_text(xlabel)
    y.set_text(ylabel)

# Read hagelslag track_step csv file into pandas DataFrame.
tracks = tdir + '/' + initial_time.strftime('track_step_NCARSTORM_d01_%Y%m%d-0000_12.csv')
if debug:
    print("reading csv file",tracks)
df = pd.read_csv(tracks, parse_dates=['Run_Date', 'Valid_Date'])
# Throw out everything except requested valid times.
df = df[df.Valid_Date == valid_time]


# Get wrfout filename
wrfout = idir + '/' + initial_time.strftime('%Y%m%d%H') + '/' + valid_time.strftime('diags_d01_%Y-%m-%d_%H_%M_%S.nc')

# Get color map, levels, and netCDF variable name appropriate for requested variable (from fieldinfo module).
info = fieldinfo.nsc[fill]
if debug:
    print("found nsc in fieldinfo.py. Using",info)
cmap = colors.ListedColormap(info['cmap'])
levels = info['levels']
fill = info['fname'][0]

if debug:
    print("About to open "+wrfout)
wrfnc = Dataset(wrfout,"r")
if fill not in wrfnc.variables:
    print("variable "+ fill + " not found")
    print("choices:", wrfnc.variables.keys())
    sys.exit(1)

# Get a 2D var from wrfout file. It has projection info.
if debug:
    print("getvar...")
cvar = getvar(wrfnc,fill)

if hasattr(cvar, 'long_name'):
    label = cvar.long_name
elif hasattr(cvar, 'description'):
    label = cvar.description

wrflat, wrflon = latlon_coords(cvar)
# get cartopy mapping object
if debug:
    print("get_cartopy...")
WRF_proj = get_cartopy(cvar)

fig = plt.figure()
if debug:
    print("plt.axes()")
ax = plt.axes(projection=WRF_proj)
ax.add_feature(cartopy.feature.STATES.with_scale('50m'), linewidth=0.4, alpha=0.6)

if cvar.min() > levels[-1] or cvar.max() < levels[0]:
    print('levels',levels,'out of range of cvar', cvar.values.min(), cvar.values.max())
    sys.exit(2)
if debug: 
    print('levels:',levels, 'cmap:', cmap.colors)

if debug:
    print("plotting filled contour",cvar.name,"...")
cfill   = ax.contourf(to_np(wrflon), to_np(wrflat), to_np(cvar), levels=levels, cmap=cmap, transform=cartopy.crs.PlateCarree() )

# Color bar
cb = plt.colorbar(cfill, ax=ax, format='%.0f', label=label+" ("+cvar.units+")", 
        shrink=0.52, orientation='horizontal')
if cb.get_ticks().size < 9:
    # ticks=levels (label every level) if there is room.
    cb.set_ticks(levels)
cb.ax.tick_params(labelsize='xx-small')
cb.outline.set_linewidth(0.5)

# Create 2 text object placeholders for spatial scale. Will be updated with each set_extent().
scale_kw = {"ha":"center","rotation_mode":"anchor","transform":ax.transAxes}
scale_xy = (ax.text(-0.01, 0.5, "", va='bottom', rotation='vertical', **scale_kw),
            ax.text(0.5, -0.01, "", va='top',  rotation='horizontal', **scale_kw))

# Special case of composite reflectivity, UH overlay
if args.fill == 'crefuh':
    max_uh = getvar(wrfnc,info['fname'][1])
    min_uh = getvar(wrfnc,info['fname'][2])
    max_uh_threshold = info['max_threshold']
    min_uh_threshold = info['min_threshold']
    print("UH max:", max_uh.max(), "UH min:", min_uh.min())
    if max_uh.max() > max_uh_threshold:
        print("Filled contour UH >",max_uh_threshold)
        # Don't use contourf if the data fall outside the levels range. You will get ValueError: 'bboxes' cannot be empty. See https://github.com/SciTools/cartopy/issues/1290
        cs1 = ax.contourf(to_np(wrflon), to_np(wrflat), to_np(max_uh), levels=[max_uh_threshold,1000], colors='black', alpha=0.3, transform=cartopy.crs.PlateCarree() )
        if debug: print("solid contour UH >",max_uh_threshold)
        cs2 = ax.contour(to_np(wrflon), to_np(wrflat), to_np(max_uh), levels=max_uh_threshold*np.arange(1,6), colors='black', linestyles='solid', linewidths=0.4, transform=cartopy.crs.PlateCarree() )
        ax.set_title(ax.get_title() + " UH>"+str(max_uh_threshold) +" "+ max_uh.units)
        # Oddly, the zero contour is plotted if there are no other valid contours
        if 0.0 in cs2.levels:
            print("uh has zero contour for some reason. Hide it")
            if debug:
                pdb.set_trace()
            for i in cs2.collections: i.remove()

    if min_uh.min() < min_uh_threshold:
        print("Filled UH contour <",min_uh_threshold)
        # Don't use contourf if the data fall outside the levels range. You will get ValueError: 'bboxes' cannot be empty. See https://github.com/SciTools/cartopy/issues/1290
        negUH1 = ax.contourf(to_np(wrflon), to_np(wrflat), to_np(min_uh), levels=[-1000, min_uh_threshold], colors='black', alpha=0.3, transform=cartopy.crs.PlateCarree() )
        if debug: print("dashed contour UH <",min_uh_threshold)
        negUH2 = ax.contour(to_np(wrflon), to_np(wrflat), to_np(min_uh), levels=min_uh_threshold*np.arange(6,1,-1), colors='black', linestyles='dashed', linewidths=0.4, transform=cartopy.crs.PlateCarree() )
        ax.set_title(ax.get_title() + " UH<"+str(-min_uh_threshold) +" "+ min_uh.units)
        if 0.0 in negUH2.levels:
            print("neg uh has a zero contour. Hide it")
            if debug:
                pdb.set_trace()
            for i in negUH2.collections: i.remove()

# Read my own county shape file.
if args.counties:
    if debug:
        print("About to draw counties")
    reader = cartopy.io.shapereader.Reader('/glade/work/ahijevyc/share/shapeFiles/cb_2013_us_county_500k/cb_2013_us_county_500k.shp')
    counties = list(reader.geometries())
    # Create custom cartopy feature that can be added to the axes.
    COUNTIES = cartopy.feature.ShapelyFeature(counties, cartopy.crs.PlateCarree())
    print("adding counties...")
    ax.add_feature(COUNTIES, facecolor="none", edgecolor='black', alpha=0.25, linewidth=0.2)

if barb:
    # Get barb netCDF variable name appropriate for requested variable (from fieldinfo module).
    info = fieldinfo.nsc[barb]
    if debug:
        print("found nsc in fieldinfo.py. Using",info)
    u,v = getvar(wrfnc, 'uvmet10', units='kt')
    skip = fieldinfo.nsc[barb]['skip']

    if args.fill ==  'crefuh': alpha=0.5
    else: alpha=1.0

    if debug: print("plotBarbs: starting barbs")
    # TODO: orient barbs with map projection (in Basemap, we use m.rotate_vector()). not important for small domain.
    if debug:
        print("barbs...")
    cs2 = ax.barbs(to_np(wrflon)[::skip,::skip], to_np(wrflat)[::skip,::skip], to_np(u)[::skip,::skip], to_np(v)[::skip,::skip], color='black', alpha=alpha, length=3.9, linewidth=0.25, sizes={'emptybarb':0.05}, transform=cartopy.crs.PlateCarree())
    ax.set_title(ax.get_title() + " wind barb (" + u.units + ")")


# Empty string placeholder for fine print in lower left corner of image.
fineprint = plt.annotate(s="", xy=(260,5), xycoords='figure pixels', fontsize=4)

for lon,lat,stepid,trackid in zip(df.Centroid_Lon, df.Centroid_Lat,df.Step_ID,df.Track_ID):

    pngfile = odir + '/' + stepid + ".png"

    string  = os.path.realpath(wrfout)
    string += "\ntrack "+trackid
    string += "\ncreated "+str(datetime.datetime.now(tz=None)).split('.')[0]

    if debug:
        fineprint.set_text(string)

    # If png already exists skip this file
    if not force_new and os.path.isfile(pngfile):
        print(pngfile + " exists. Skipping. Use --force_new option to override.")
        continue
    x, y = WRF_proj.transform_point(lon, lat, cartopy.crs.PlateCarree()) # Transform lon/lat to x and y (in meters) in WRF projection.
    ax.set_extent([x-padding[0]*1000., x+padding[1]*1000., y-padding[2]*1000., y+padding[3]*1000.], crs=WRF_proj)

    # Update axes labels.
    change_scale(scale_xy)

    plt.savefig(pngfile, dpi=175)
    print('created ' + os.path.realpath(pngfile))

if debug: pdb.set_trace()
plt.close(fig)
print("Run this command to create a montage")
print("montage -crop 465x465+340+95 -geometry 70% -tile 5x4 d01*png t.png")
