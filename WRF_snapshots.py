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
parser = argparse.ArgumentParser(description = "Plot WRF and SPC storm reports")
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
info = fieldinfo.fieldinfo[fill]
if debug:
    print("found fieldinfo in fieldinfo.py. Using",info)
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
cvar_units_str = cvar.units

wrflat, wrflon = latlon_coords(cvar)
# get cartopy mapping object
if debug:
    print("get_cartopy...")
WRF_proj = get_cartopy(cvar)
if debug:
    pdb.set_trace()

fig = plt.figure()
if debug:
    print("plt.axes()")
ax = plt.axes(projection=WRF_proj)
ax.add_feature(cartopy.feature.STATES.with_scale('50m'), linewidth=0.45, alpha=0.75)

if cvar.min() > levels[-1] or cvar.max() < levels[0]:
    print('levels',levels,'out of range of cvar', cvar.values.min(), cvar.values.max())
    sys.exit(2)
if debug: 
    print('levels:',levels, 'cmap:', cmap.colors)

if debug:
    print("plotting filled contour...")
cfill   = ax.contourf(to_np(wrflon), to_np(wrflat), to_np(cvar), levels=levels, cmap=cmap, transform=cartopy.crs.PlateCarree() )

# Special case of composite reflectivity, UH overlay
if args.fill == 'crefuh':
    uh = getvar(wrfnc,info['fname'][1])
    print("Overlay UH",uh.max())
    cs1 = ax.contourf(to_np(wrflon), to_np(wrflat), to_np(uh), levels=[100,1000], colors='black', alpha=0.3, transform=cartopy.crs.PlateCarree() )
    cs2 = ax.contour(to_np(wrflon), to_np(wrflat), to_np(uh), levels=[100], colors='black', linewidths=0.5, transform=cartopy.crs.PlateCarree() )
    # for some reason the zero contour is plotted if there are no other valid contours
    # are there some small negatives due to regridding? No.
    if 0.0 in cs2.levels:
        print("uh has zero contour for some reason. Hide it")
        for i in cs2.collections:
            i.remove()

# Read my own county shape file.
if args.counties:
    if debug:
        print("About to draw counties")
        pdb.set_trace()
    reader = cartopy.io.shapereader.Reader('/glade/work/ahijevyc/share/cb_2013_us_county_500k/cb_2013_us_county_500k.shp')
    counties = list(reader.geometries())
    # Create custom cartopy feature that can be added to the axes.
    COUNTIES = cartopy.feature.ShapelyFeature(counties, cartopy.crs.PlateCarree())
    print("adding counties...")
    ax.add_feature(COUNTIES, facecolor="none", edgecolor='black', alpha=0.25, linewidth=0.2)

if barb:
    # Get barb netCDF variable name appropriate for requested variable (from fieldinfo module).
    info = fieldinfo.fieldinfo[barb]
    if debug:
        print("found fieldinfo in fieldinfo.py. Using",info)
    ustr, vstr = info['fname']
    u = getvar(wrfnc, ustr)
    v = getvar(wrfnc, vstr)
    skip = 4

    if args.fill ==  'crefuh': alpha=0.5
    else: alpha=1.0

    if debug: print("plotBarbs: starting barbs")
    # TODO: orient barbs with map projection (in Basemap, we use m.rotate_vector())
    if debug:
        print("barbs...")
    cs2 = ax.barbs(to_np(wrflon)[::skip,::skip], to_np(wrflat)[::skip,::skip], to_np(u)[::skip,::skip], to_np(v)[::skip,::skip], color='black', alpha=alpha, length=3.9, linewidth=0.25, sizes={'emptybarb':0.05}, transform=cartopy.crs.PlateCarree())


# Color bar
cb = plt.colorbar(cfill, ax=ax, format='%.0f', label=label+" ("+cvar_units_str+")", 
        shrink=0.55, orientation='horizontal')
if cb.get_ticks().size < 9:
    # ticks=levels (label every level) if there is room.
    cb.set_ticks(levels)
cb.ax.tick_params(labelsize='xx-small')
cb.outline.set_linewidth(0.5)


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
    plt.savefig(pngfile, dpi=175)
    print('created ' + os.path.realpath(pngfile))

if debug: pdb.set_trace()
plt.close(fig)
print("Run this command to create a montage")
print("montage -crop 490x490+328+80 -geometry 70% -tile 5x4 d01*png t.png")
