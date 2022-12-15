import cartopy
import cartopy.geodesic
import datetime
import glob # used to locate stormEvents files 
import logging
import matplotlib.pyplot as plt
from   metpy.units import units # used to normalize polarplot range
import numpy as np
import os # for basename
import pandas as pd
from   pandas.api.types import CategoricalDtype
import pdb
import pytz # Time zone-aware datetimes
import requests # for stormEvents()
from   scipy import spatial
import scipy.ndimage as ndimage
from scipy.stats import circmean # To average longitudes
from scipy.stats import gaussian_kde
import seaborn as sns
import sqlite3
import sys # for stderr output
import xarray

def getTCTOR(rename=True):
    # Tropical Cyclone Tornado (TCTOR) database from Roger Edwards (SPC)
    ifile = "/glade/work/ahijevyc/share/SPC/TCTor95-18.xls - Full List.csv"
    REdf = pd.read_csv(ifile, delimiter=',', parse_dates=[["Year","UTC-Mo.","UTC-Date","UTC time"]], 
            index_col=0, skiprows=1, header=[0], nrows=1506) # skipfooter=23)
    # skipfooter=23 could be used, but 'c' engine doesn't handle it. 'c' engine is faster than 'python' engine.
    REdf = REdf.tz_localize(tz='UTC')

    if rename:
        REdf.index.set_names('datetime', inplace=True)
        # Roger's columns have nice, descriptive names, but
        # rename some of Roger's columns to match terse SPC database.
        RE2SPC = {
                "LAT-start (path)" : "slat",
                "LON-start (path)" : "slon",
                "LAT-end (path)"   : "elat",
                "LON-end (path)"   : "elon"
                }

        REdf = REdf.rename(columns=RE2SPC)


    # Fix probable typo. "ST" should be "TS"
    REdf.loc[REdf["TC Cat"] == "ST","TC Cat"] = "TS"
    # Convert to dtype="category" - could do this at read_csv step, but then "ST" exists as a category.
    cat_type = CategoricalDtype(categories=["N","TD","TS","H","MH"], ordered=True)
    REdf["TC Cat"] = REdf["TC Cat"].astype(cat_type)


    return REdf


def RyanSobash(start=None, end=None, event_type="torn"):
    sobash_db = "/glade/u/home/sobash/2013RT/REPORTS/reports_v20200626.db"
    conn = sqlite3.connect(sobash_db)
    sqltable = "reports_" + event_type
    # Could apply a datetime range here (WHERE datetime BETWEEN yyyy/mm/dd hh:mm:ss and blah), but converting from UTC to CST is tricky.
    sqlcommand = "SELECT * FROM "+sqltable
    sql_df = pd.read_sql_query(sqlcommand, conn, parse_dates=['datetime'])
    conn.close()
    # Add 6 hours to datetime. This converts to UTC.
    sql_df["datetime"] = sql_df["datetime"] + pd.to_timedelta(6, unit='h')
    # make it aware of its UTC timezone.
    sql_df["datetime"] = sql_df["datetime"].dt.tz_localize(pytz.UTC)
    sql_df = sql_df[(sql_df.datetime >= start) & (sql_df.datetime < end)]
    sql_df = sql_df.sort_values(by=["datetime","slon","slat"])# did this with SPC too ("time" instead of "datetime")

    sql_df["dbfile"] = sobash_db
    logging.debug(f"From Ryan's SQL database {sobash_db}")
    logging.debug(sqlcommand)
    logging.debug(sql_df.to_string())

    return sql_df, sobash_db # Yes, sobash_db is in the dbfile column of sql_df, but sql_df may have zero rows, so return sobash_db separately.

def spc_lsr_filename(event_type, latestyear=2020):
    assert event_type in ["wind","hail","torn"]
    # Get wind, hail, or tornado local storm reports (LSR). If LSR do not exist locally, download from SPC.
    # Cache in TMPDIR, which by default is /glade/scratch/$USER/temp/.

    # Input
    # event_type: "wind", "hail", or "torn"
    # latestyear: most recent (4-digit) year in SPC dataset. Part of SPC filename. Update when SPC releases new year.

    import urllib.request
    idir = os.getenv("TMPDIR", "/glade/scratch/"+os.getenv("USER")+"/temp")
    logging.debug(f"spc.spc_lsr_filename: idir={idir}")

    if event_type=='torn':
        filename = idir + f"/1950-{latestyear}_actual_tornadoes.csv"
    else:
        filename = idir + f"/1955-{latestyear}_"+event_type+".csv.zip"

    logging.debug(f"spc.spc_lsr_filename: local filename={filename}")
    if not os.path.exists(filename):
        url = "https://www.spc.noaa.gov/wcm/data/" + os.path.basename(filename)
        logging.debug(f"spc.spc_lsr_filename: local file not found. downloading {url}")
        urllib.request.urlretrieve(url, filename)

    nominal_last_time = datetime.datetime(latestyear+1,1,1,0,0,0,0,pytz.UTC) 
    return filename, nominal_last_time


def get_storm_reports( start = datetime.datetime(2016,6,10,tzinfo=pytz.UTC), end = datetime.datetime(2016,7, 1,tzinfo=pytz.UTC), 
        event_types = ["torn", "wind", "hail"], latestyear = 2020, RyanSobashSanityCheck=False):

    # Return a DataFrame with local storm reports (LSRs) downloaded from SPC.
    # Choices are tornado, hail, and/or wind.
    assert isinstance(event_types,list), "expected event_types to be a list"

    # INPUT
    #   start - start of time window, includes start. timezone aware datetime
    #   end   - end of time window, includes end. timezone aware datetime
    #   event_types - list of event types
    #   latestyear - most recent year in SPC file. 4-digit year
    # OUTPUT
    #   rpts - DataFrame with time, location, and description of event (LSR)


    logging.debug(f"get_storm_reports: start:{start}")
    logging.debug(f"get_storm_reports: end:{end}")
    logging.debug(f"get_storm_reports: event types:{event_types}")


    # Create one DataFrame with all requested LSR event types.
    all_rpts = pd.DataFrame()
    for event_type in event_types:
        # Locate data file and nominal last time in data file.
        rpts_file, nominal_last_time = spc_lsr_filename(event_type, latestyear=latestyear)

        # Make sure the requested last time isn't later than the nominal last time.
        if end > nominal_last_time:
            logging.error(f"spc.get_storm_reports({event_type}): requested end time {end} is later than nominal last time in {rpts_file} {nominal_last_time}")
            logging.error(f"Do you need to download a new {event_type} database from SPC?")
            sys.exit(1)

        # csv format described in http://www.spc.noaa.gov/wcm/data/SPC_severe_database_description.pdf
        # SPC storm report files downloaded from http://www.spc.noaa.gov/wcm/#data to 
        # cheyenne:/glade/work/ahijevyc/share/ Mar 2019.
        # Multi-year zip files have headers; pre-2016 single-year csv files have no header. 

        dtype = {
                "om": np.int64,
                "yr": np.int32,
                "mo": np.int32,
                "dy": np.int32,
                "date": str,
                "tz": np.int32,
                "st": str,
                "stf": np.int64, # State FIPS number. some Puerto Rico codes are incorrect
                "stn": np.int64,  # State number - number of this tornado in this state in this year
                "mag": np.float64, # you might think there is "sz" for hail and "f" for torn, but all "mag"
                "inj": np.int64,
                "fat": np.int64,
                "loss": np.float64,
                "closs": np.float64,
                "slat": np.float64,
                "slon": np.float64,
                "elat": np.float64,
                "elon": np.float64,
                "len": np.float64,
                "wid": np.float64,
                "ns": np.int64,
                "sn": np.int64,
                "sg": np.int64,
                "f1": np.int64,
                "f2": np.int64,
                "f3": np.int64,
                "f4": np.int64,
                "mt": str,
                "fc": np.int64,
                }

        # Unify "date" and "time" to make a naive datetime object (datetime with no timezone information).
        # The unified column is called "date_time".
        rpts = pd.read_csv(rpts_file, parse_dates=[['date','time']], dtype=dtype, infer_datetime_format=True)
        logging.debug(f"input file: {rpts_file}. read {len(rpts)} lines")

        rpts["event_type"] = event_type
        rpts["source"] = rpts_file # Used to take basename of this. But why dispose dirname information?

        # -9 = unknown tornado F-scale
        # Change -9 to NaN
        rpts["mag"].replace(to_replace=-9, value = np.nan, inplace=True)

        # "Significant" is defined as: tornadoes rated EF2 or greater, thunderstorm wind gusts of hurricane force (74 mph) or higher, or hail 2 inches or larger in diameter.
        rpts["significant"] = ((event_type == "torn") & (rpts.mag >= 2.)) | ((event_type == "wind") & (rpts.mag >= 65.)) | ((event_type == "hail") & (rpts.mag >= 2.)) 
        largehail = rpts["significant"] & (rpts["event_type"] == "hail")
        rpts.loc[largehail,"event_type"] = "large hail"
        highwind = rpts["significant"]  & (rpts["event_type"] == "wind")
        rpts.loc[highwind, "event_type"] = "high wind"
        sigtorn = rpts["significant"]  & (rpts["event_type"] == "torn")
        rpts.loc[sigtorn, "event_type"] = "sigtorn"



        # Derive timezone-aware datetime "time" from "date_time" and "tz" columns. Convert to UTC.
        # According to www.spc.noaa.gov/wcm/data/SPC_severe_database_description.pdf,
        # all date_times except for tz ?=unknown and 9=GMT were converted to 3=CST. 
        # But there are several tz=0 and tz=6. tz=6 is MDT in the NECI Storm Events database.

        # MDT is equivalent to CST. Therefore, change tz=6 (MDT) to tz=3 (CST).
        # Wrote to SPC Apr 1 2019 about fixing these lines.
        MDT = rpts['tz'] == 6
        logging.debug(f"spc.get_storm_reports(): found {MDT.sum()} {event_type} in MDT")
        logging.debug(rpts.loc[MDT, ['om','date_time','tz','event_type', 'source']])
        logging.debug("changing tz from 6 to 3 because CST=MDT")
        rpts.loc[MDT, "tz"] = 3

        # Convert GMT(UTC) date_times to CST
        GMT = rpts.tz == 9
        logging.debug(f"spc.get_storm_reports(): found {GMT.sum()} {event_type} in GMT")
        rpts.loc[GMT, "date_time"] = rpts.loc[GMT, "date_time"] - datetime.timedelta(hours=6)
        rpts.loc[GMT, "tz"] = 3

        # Don't know what to with date_times with unknown time zones. They're treated like CST when converted to UTC below. 
        unknown_tz = rpts['tz'] == 0
        logging.debug(f"spc.get_storm_reports(): found {unknown_tz.sum()} {event_type} in unknown time zone")
        logging.debug(rpts.loc[unknown_tz, ['om','date_time','tz','event_type', 'source']])

        """
        Wrote to SPC Apr 1 2019 about fixing tz!=3 lines.
        Email20190401 PatrickMarsh took over database in 2017 and has no record or documentation as to what those timezones are. 
        Each year I append new information to the end of the old information, so timezones will continue to exist as is until I can learn what those time zones are. 
        take a look at the NCEI version of storm data. They may have information I do not.
        
        When timezone = 0, it is 'UNK' (unknown?) in the NCEI Storm Events database.
        When timezone = 6, it is 'MDT' (Mountain Daylight Time?) in the NCEI Storm Events database.
        
        LSR with tz=0 or tz=6, followed by the zone (if any) according to NCEI Storm Events database:
                om       yr_mo_dy_time  tz zone
        2507   245 1956-06-01 11:33:00   0 UNK
        2855    89 1957-04-02 23:45:00   0 UNK
        9569   158 1967-04-21 12:33:00   0 UNK
        13409  264 1972-05-13 18:08:00   0 UNK
        15792  804 1974-08-13 15:03:00   0 UNK
        20640  458 1980-06-04 16:30:00   0 UNK
        22101  271 1982-05-11 14:25:00   0 UNK
        24007  200 1984-04-26 19:32:00   0 CST
        8145   216 1965-05-05 14:45:00   6 MDT
        25899  501 1986-07-01 22:15:00   6 MDT
        26054  656 1986-09-04 18:55:00   6 MDT
        28793  419 1990-05-24 15:00:00   6 MDT
        28797  422 1990-05-24 16:00:00   6 MDT
        28810  433 1990-05-24 18:33:00   6 MDT
        29192  815 1990-06-27 20:00:00   6 MDT
        29232  855 1990-07-05 21:10:00   6 MDT
        29347  971 1990-08-15 18:30:00   6 MDT
        33262  151 1994-04-22 18:06:00   6 MDT
        33557  446 1994-05-31 15:00:00   6 MDT
        33572  461 1994-06-06 14:40:00   6 MDT
        33574  463 1994-06-06 15:00:00   6 MDT
        33585  474 1994-06-07 14:47:00   6 MDT
        33586  476 1994-06-07 15:57:00   6 MDT
        33587  477 1994-06-07 16:10:00   6 MDT
        33589  478 1994-06-07 16:35:00   6 MDT
        33592  482 1994-06-07 18:50:00   6 not in Storm Events database
        33783  672 1994-06-29 15:45:00   6 MDT
        33834  722 1994-07-06 18:17:00   6 MDT
        33855  744 1994-07-12 14:30:00   6 MDT
        33886  775 1994-07-18 15:30:00   6 MDT
        33887  776 1994-07-18 16:00:00   6 MDT
        33888  777 1994-07-18 16:25:00   6 MDT
        33889  778 1994-07-18 16:40:00   6 MDT
        33890  779 1994-07-18 16:55:00   6 not in Storm Events database
        33891  780 1994-07-18 16:55:00   6 not in Storm Events database
        33892  781 1994-07-18 17:00:00   6 MDT
        """            

        # Now all date_times, except for ?=unknown, were converted to 3=CST.
        # Convert to UTC by adding 6 hours.
        rpts["time"] = rpts['date_time'] + pd.to_timedelta(6, unit='h')
        # Make time-zone aware datetime object (UTC) "time" . 
        rpts["time"] = rpts["time"].dt.tz_localize(pytz.UTC)

        # Apply requested time window start <= time <= end
        # Used to say rpts.time < end. But if rpts.time == end it would not get included. 
        time_window = (rpts.time >= start) & (rpts.time <= end)
        rpts = rpts[time_window]

        rpts = rpts.sort_values(by=["time","slon","slat"])
        logging.debug(f"found {len(rpts)} {event_type} reports")


        if RyanSobashSanityCheck:
            # Verify I get the same thing as Ryan Sobash's sqlite3 database
            sql_df, RyanSobash_file = RyanSobash(start=start, end=end, event_type=event_type)

            if rpts.empty and sql_df.empty:
                logging.info(f"No {event_type} reports in spc or Ryan's database")
                pass
            elif len(sql_df) != len(rpts): # See if they have the same number of rows
                logging.warning(f"SPC has {len(rpts)} {event_type} reports, but Ryan's SQL database has {len(sql_df)}.")
                epoch1 = os.path.getmtime(rpts_file)
                epoch2 = os.path.getmtime(RyanSobash_file)
                if epoch2 > epoch1:
                    logging.warning("Ryan's database modified more recently but it may have duplicate reports.")
                logging.info(f"Mod date of SPC reports file:    {datetime.datetime.fromtimestamp(epoch1).strftime('%c')}")
                logging.info(f"Mod date of Ryan's SQL database: {datetime.datetime.fromtimestamp(epoch2).strftime('%c')}")
            elif not all(sql_df["datetime"].to_numpy() == rpts["time"].to_numpy()): # Times don't all match. ignore index
                logging.info("Ryan's database has same # of reports but times aren't equal. Can't easily compare the lat and lons")
                print(f"SPC\n{rpts}")
                print(f"Ryan's\n{sql_df}")
            else:
                # See if they have the same locations
                same_columns = ["slat", "slon", "elat", "elon"]
                # ignore index (which has no real meaning and differs between datasets)
                mismatch = (sql_df.reset_index()[same_columns] != rpts.reset_index()[same_columns]).any(axis="columns")
                if mismatch.any(): # any rows and any columns
                    logging.warning("SPC locations don't match Ryan's SQL database")
                    mine = rpts.loc[mismatch,same_columns]
                    his  = sql_df.loc[mismatch,same_columns]
                    print("SPC minus Ryan")
                    difference = mine-his
                    print(difference)
                    max_abs_difference = difference.abs().max().max()
                    logging.warning(f"max abs difference {max_abs_difference}")
                    if max_abs_difference < 0.000001:
                        logging.info("who cares about such a small difference?")
                    elif max_abs_difference < 0.1:
                        pass
                    else:
                        logging.error("That's concerning")
                        pdb.set_trace()
                        sys.exit(1)

        all_rpts = all_rpts.append(rpts) # Append this storm report type

    return all_rpts

def symbol_dict(scale=1):
    # Color, size, marker, and label of wind, hail, and tornado storm reports
    d = {
            "wind":       {"c" : 'blue',  "s": 8*scale, "marker":"s", "label":"Wind"},
            "high wind":  {"c" : 'black', "s":12*scale, "marker":"s", "label":"Wind/65kt+"},
            "hail":       {"c" : 'green', "s":12*scale, "marker":"^", "label":"Hail"},
            "large hail": {"c" : 'black', "s":16*scale, "marker":"^", "label":'Hail/2"+'},
            "torn":       {"c" : 'red',"s": 6*scale, "marker":"v", "label":"Torn"}, # pink was okay too
            "sigtorn":    {"c" : 'red',"s":12*scale, "marker":"v", "label":"Torn/EF2+"}
         }
    for k in d:
        d[k]["edgecolors"]="black"
        d[k]["linewidths"] = 0.001
    d["F-sum"] = d["torn"].copy()
    return d

# Tried including units but DataArray.plot.contour chokes on the levels with pint.error
kdelevels = {
        "torn":np.arange(0,4,0.4)*1e-4,
        "F-sum":np.arange(0,4,0.4)*1e-4,
        "wind":np.arange(0,5,0.5)*1e-4,
        "hail":np.arange(0,5,0.5)*1e-4
        }



def plotgridded(storm_reports, ax, gridlat2D=None, gridlon2D=None, scale=1, sigma=1, alpha=0.5, event_types=["wind", "high wind", "hail", "large hail", "torn"]):


    # Default dx and dy of 81.2705km came from https://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html#GRID211
    if storm_reports.empty:
        logging.debug("spc.plotgridded(): storm reports DataFrame is empty. data_gridded will be all zeros.")

    storm_rpts_gridded = {}

    for event_type in event_types:
        logging.debug("looking for "+event_type)
        xrpts = storm_reports[storm_reports.event_type == event_type]
        print("plot",len(xrpts),event_type,"reports")
        # Used to skip to next iteration if there were zero reports, but you want grid even if it is all zeros.
        obslons, obslats = xrpts.slon.values, xrpts.slat.values

        grid_points = list(zip(gridlon2D.ravel(),gridlat2D.ravel()))# tried using zip but in python 3 this returns a 0-dimensional zip object, not 2 elements
        tree = spatial.KDTree(grid_points) 

        data_gridded = np.zeros_like(gridlon2D).flatten()
        if len(obslons) > 0:
            dist, indices = tree.query(list(zip(obslons,obslats)))
            data_gridded[indices] = 1

        data_gridded = data_gridded.reshape(gridlon2D.shape)

        if sigma > 0:
            logging.debug("about to run Gaussian smoother")
            # I'm uncomfortable doing this in map space but it is fast
            data_gridded = ndimage.gaussian_filter(data_gridded, sigma=sigma)

        storm_rpts_gridded[event_type] = data_gridded

        if False:
            # TODO: fix extent so it is plotted correctly
            color = symbol_dict()[event_type]["c"]
            # get the color of the symbol, tack on an "s" and use that as the color map name.
            if color=="black": # there is no "blacks" colortable
                color="grey"
            color = color.capitalize()
            # add gridded storm events array to dictionary

            img = ax.imshow(np.ma.masked_less(data_gridded,0.001), extent=(gridlon2D.min(),gridlon2D.max(),gridlat2D.min(),gridlat2D.max()),
                    transform=cartopy.crs.PlateCarree(), cmap=plt.get_cmap(color+"s"), label=event_type, alpha=0.85)

    return storm_rpts_gridded

def centroid_polar(theta_deg, r):
    # locate centroid of this event type
    east = r * np.sin(theta_deg)
    north = r * np.cos(theta_deg)
    logging.debug(f"spc.centroid_polar(): east {east}")
    logging.debug(f"spc.centroid_polar(): north {north}")
    east = east.mean()
    north = north.mean()
    logging.debug(f"east mean {east:.2f} north mean {north:.2f}")
    az = np.degrees(np.arctan2(east,north))
    r = np.sqrt(east**2 + north**2)
    logging.debug(f"spc.centroid_polar(): x,y {east:7.2f},{north:7.2f}  az,r {az:5.1f},{r:6.2f}")

    return az, r

def combine_significant(storm_reports):
    # combine torn and sigtorn, wind and "high wind", hail and "large hail"
    sigtorn = storm_reports["event_type"] == "sigtorn"
    storm_reports.loc[sigtorn, "event_type"] = "torn"
    sigwind = storm_reports["event_type"] == "high wind"
    storm_reports.loc[sigwind, "event_type"] = "wind"
    sighail = storm_reports["event_type"] == "large hail"
    storm_reports.loc[sighail, "event_type"] = "hail"
    return storm_reports

def fill_NaNs_with_neighbors2d(array2d):
    # Fill NaNs with neighbors
    # Input DataArray
    # convert to ndarray
    values = array2d.values
    values = pd.DataFrame(values)
    # operate on DataFrame
    values = values.interpolate(limit_direction='both') # goofy around NARR edges
    if values.isnull().all().any(): # if any columns are all null
        logging.warning("***Found nan in filled value array. trying to interpolate across columns***") # Think this is fine but not sure.
        # Happens when you zoom in a lot.
        values = values.interpolate(axis='columns')
        if values[0].isnull().all(): # if column 0 is still all null, backfill from neighboring column
            values.fillna(method="backfill", axis="columns", inplace=True)
            logging.warning("***Backfilled missing data in column 0***")
    # Output DataArray
    array2d.values = values
    return array2d

def fill_NaNs_with_neighbors(values):        
    if len(values.shape) == 2:
        return fill_NaNs_with_neighbors2d(values)
    if len(values.shape) == 3:
        for i, aa in enumerate(values):
            aa = fill_NaNs_with_neighbors2d(aa)
            values[i] = aa.values
    return values


def histogram2d_weighted(bearing, dist_from_center, azbins, rbins, data):
    bins = [azbins, rbins]
    bearing = bearing.values.flatten()
    dist_from_center = dist_from_center.values.flatten()
    data = data.metpy.dequantify() # move units to attrs dict so we may assign them to new DataArray.
    n, _, _ = np.histogram2d(bearing, dist_from_center, bins=bins)
    drs = np.diff(rbins) # bin widths
    dazs = np.diff(azbins)
    assert np.all(drs == drs[0]), "range bins not equal width"
    assert np.all(dazs == dazs[0]), "azimuth bins not equal width"
    range_center1D, theta_center1D = drs[0]/2+rbins[:-1], dazs[0]/2+azbins[:-1]
    isvector = 'uv' in data.coords
    if isvector:
        weightedh2 = np.zeros((2, len(bins[0])-1, len(bins[1])-1))
        weightedh2[0], _, _ = np.histogram2d(bearing, dist_from_center, bins=bins, weights=data.isel(uv=0).values.flatten())
        weightedh2[1], _, _ = np.histogram2d(bearing, dist_from_center, bins=bins, weights=data.isel(uv=1).values.flatten())
        coords={"uv": data.coords["uv"], "azimuth":theta_center1D, "range":range_center1D}
    else:
        weightedh2, _, _ = np.histogram2d(bearing, dist_from_center, bins=bins, weights=data.values.flatten())
        coords={"azimuth":theta_center1D, "range":range_center1D}
    n = np.ma.masked_where(n == 0,n)# avoid RuntimeWarning: invalid value encountered in true_divide
    da = xarray.DataArray(data = weightedh2/n, coords=coords, dims=coords.keys(), attrs=data.attrs, name=data.name) 
    # Fill NaNs with neighbors
    da = fill_NaNs_with_neighbors(da)
    da = da.metpy.quantify() # move units from attrs dict back to data array (make it a pint quantity)
    return da


def polarkde(originlon, originlat, storm_reports, ax, azbins, rbins, spc_td, ds=20*units.km, 
        zero_azimuth=0*units.deg, normalize_range_by=None, add_colorbar=False):
    # Return dictionary where key/value pairs are { event_type : kde on cartesian grid }
    # Input
    # azbins, rbins from NARR_composite.py
    # spc_td SPC reports time window
    # ds [km] spacing of Cartesian mesh grid on which smoothing is performed 
    storm_rpts_kde = {}
    if storm_reports.empty:
        logging.debug("spc.polarkde(): storm reports DataFrame is empty. Returning None")
        return storm_rpts_kde

    geo = cartopy.geodesic.Geodesic()
    # Use .m because geo.inverse does not work with quantities.
    n3 = geo.inverse((originlon.m, originlat.m), np.column_stack((storm_reports["slon"], storm_reports["slat"])))
    n3 = np.asarray(n3) # convert cartopy MemoryView to ndarray
    storm_reports["range"] = n3[:,0]/1000. # geo.inverse returns meters; convert to km
    if normalize_range_by:
        storm_reports["range"] /= normalize_range_by
    storm_reports["heading"] = n3[:,1]  # in deg
    storm_reports["heading"] = (storm_reports["heading"] - zero_azimuth.m + 720) % 360 # remove units from zero_azimuth, or it assumes other variables are radians.
    rmax = ax.get_ylim()[1] * units.km
    inrange = storm_reports["range"].values * units.km  < rmax # have to use .values because units won't stick to pandas Series.
    storm_reports = storm_reports[inrange]
    logging.debug(f"spc.polarplot(): found {inrange.sum()} reports inside axis range")

    storm_reports = storm_reports[storm_reports["range"] < rbins.max()] # Throw out reports beyond max range
    storm_reports = storm_reports.copy() # Avoid SettingWithCopyWarning
    for event_type, xrpts in combine_significant(storm_reports).groupby("event_type"):    
        logging.debug(f"spc.polarkdeplot(): found {len(xrpts)} {event_type} reports")
        if len(xrpts) < 3:
            logging.warning(f"not enough pts for {event_type} kdeplot")
            continue

        # convert to Cartesian and apply Gaussian kde
        rptx = xrpts["range"] * np.cos(np.radians(xrpts["heading"]))
        rpty = xrpts["range"] * np.sin(np.radians(xrpts["heading"]))
        n = len(xrpts)
        X, Y = np.mgrid[-rmax:rmax+ds:ds, -rmax:rmax+ds:ds] 
        bearing = pd.DataFrame(np.degrees(np.arctan2(Y,X))) # Make DataFrame because histogram2d_weighted expects a .values attribute
        bearing[bearing < 0] = bearing[bearing < 0] + 360 # azbins is 0-360 but bearing was -180 to +180
        dist_from_center = pd.DataFrame(np.sqrt(X**2 + Y**2))
        positions = np.vstack([X.ravel(), Y.ravel()])
        if event_type == "torn":
            logging.debug("weighting torn rpts by F-sum (EF magnitude plus one)")
            weights = xrpts["mag"].fillna(value=0) + 1 # F-sum McCaul 1991
            event_type = "F-sum"
        weights = None
        kernel = gaussian_kde(np.vstack([rptx,rpty]), weights=weights)
        Z = np.reshape(kernel(positions).T, X.shape)
        Z = xarray.DataArray(n * Z  / units.km**2 / (2*spc_td.total_seconds() * units.seconds)) # histogram2d_weighted expects a DataArray with units.
        Z = Z.metpy.convert_units("1/km**2/day")
        # convert Cartesian grid Z to polar coordinates
        pkde = histogram2d_weighted(bearing, dist_from_center, azbins, rbins, Z)
        pkde["azimuth"] = np.radians(pkde.azimuth) # Polar Axes are in radians not degrees.
        colors = symbol_dict()[event_type]["c"]
        polarc = pkde.plot.contour(x="azimuth",y="range", ax=ax, colors=colors, linewidths=0.5, levels=kdelevels[event_type],
                add_colorbar=add_colorbar, cbar_kwargs={"shrink":0.5,"pad":0.09} if add_colorbar else None)

        cb = None
        if add_colorbar:
            cb = polarc.colorbar
            cb.formatter.set_powerlimits((-2,2))
            cb.update_ticks()
            cb.ax.yaxis.offsetText.set(size='xx-small')
            cb.ax.yaxis.offsetText.set_horizontalalignment("left")
            cb.set_label(event_type + " " + cb.ax.yaxis.get_label().get_text(),fontsize="xx-small")
            cb.ax.tick_params(labelsize='xx-small')
            
        ax.set_xlabel('')
        ax.set_ylabel('')
        storm_rpts_kde[event_type] = pkde # thought about QuadContourSet polarc, but you can't back out data from it.
    return storm_rpts_kde


def polarplot(originlon, originlat, storm_reports, ax, zero_azimuth=0*units.deg, add_legend=False, normalize_range_by=None, scale=1, alpha=0.8,
        legend_title=None):
    # Return storm_reports DataFrame with additional "range" and "heading" columns

    storm_reports["range"] = None
    storm_reports["heading"] = None
    if storm_reports.empty:
        logging.debug("spc.polarplot(): storm reports DataFrame is empty.")
        return storm_reports


    # Color, size, marker, and label of wind, hail, and tornado storm reports
    kwdict = symbol_dict(scale=scale)

    geo = cartopy.geodesic.Geodesic()
    # Use .m because geo.inverse does not work with quantities.
    n3 = geo.inverse((originlon.m, originlat.m), np.column_stack((storm_reports["slon"], storm_reports["slat"])))
    n3 = np.asarray(n3) # convert cartopy MemoryView to ndarray
    storm_reports["range"] = n3[:,0]/1000. # geo.inverse returns meters; convert to km
    if normalize_range_by:
        storm_reports["range"] /= normalize_range_by
    storm_reports["heading"] = n3[:,1]  # in deg
    storm_reports["heading"] = (storm_reports["heading"] - zero_azimuth.m + 720) % 360 # if zero_azimuth has units, it assumes radians for numbers without units.

    # Filter out points beyond the max range of axis
    assert not ax.have_units(), 'spc.polarplot() found axes units. Assumed yaxis has no attached units, but is km'
    rmax = ax.get_ylim()[1] * units.km
    inrange = storm_reports["range"].values * units.km  < rmax # have to use .values because units won't stick to pandas Series.
    storm_reports = storm_reports[inrange]
    logging.debug(f"spc.polarplot(): found {inrange.sum()} reports inside axis range")

    ax.set_autoscale_on(False) # Don't rescale the axes with far-away reports (thought unneeded after filtering out pts beyond rmax, but symbol near maximum range autoscales axis to larger range.)
    for event_type, xrpts in storm_reports.groupby("event_type"):
        logging.debug(f"spc.polarplot(): found {len(xrpts)} {event_type} reports")
        kwdict[event_type]["label"] += f" ({len(xrpts)})"
        # Feed axes.scatter magnitudes, not quantities. Quantities are clever but axes.scatter uses their units as axes labels. Not good for polar plot.
        # Also, if I use quantities, the 2nd cfill plot errors out deep in matplotlib:     Nx = X.shape[-1] AttributeError: 'list' object has no attribute 'shape'
        storm_rpts_plot = ax.scatter(np.radians(xrpts["heading"]), xrpts["range"], alpha = alpha, **kwdict[event_type])

        az, r = centroid_polar(xrpts["heading"].values*units.deg, xrpts["range"].values*units.km)

    ax.set_xlabel('') # in case you figure out how to pass quantities to ax.scatter()
    ax.set_ylabel('')

    if add_legend:
        storm_rpt_legend_kw = dict(fontsize=4.8, bbox_to_anchor=(0.8, -0.01), loc='upper left', borderaxespad=0, frameon=False, title_fontsize='xx-small')
        storm_report_legend = ax.legend(title=legend_title, **storm_rpt_legend_kw)

    return storm_reports

def plot(storm_reports, ax, scale=1, drawrange=0, alpha=0.5, colorbyfreq=False):

    storm_rpts_plots = {}
    if storm_reports.empty:
        # is this the right thing to return? what about empty list []? or rpts?
        logging.debug("spc.plot(): storm reports DataFrame is empty. Returning")
        return storm_rpts_plots

    # Color, size, marker, and label of wind, hail, and tornado storm reports
    kwdict = symbol_dict(scale=scale)

    for event_type, xrpts in storm_reports.groupby("event_type"):
        logging.debug(f"plot {len(xrpts)} {event_type} reports")
        kwdict[event_type]["label"] += " (%d)" % len(xrpts)
        lons, lats = xrpts.slon.values, xrpts.slat.values
        if colorbyfreq:
            # check for multiple reports at exact same coordinates (different times, though). Draw differently (hotter color, or bigger?)
            freq = xrpts.groupby(["slon","slat"]).size()
            del(kwdict[event_type]["c"])
            del(kwdict[event_type]["s"])
            lons = [x[0] for x in freq.index]
            lats = [x[1] for x in freq.index]
            storm_rpts_plot = ax.scatter(lons, lats, c=freq.values, edgecolors="None", **kwdict[event_type], transform=cartopy.crs.PlateCarree()) 
            for (lon, lat), c in freq.iteritems():
                if c>1:
                    ax.text(lon, lat, c, transform=cartopy.crs.PlateCarree(), fontsize=6, va="center", ha="center", zorder=storm_rpts_plot.get_zorder()+1) 
        else:
            storm_rpts_plot = ax.scatter(lons, lats, alpha = alpha, **kwdict[event_type],
                    transform=cartopy.crs.PlateCarree()) # ValueError: Invalid transform: Spherical scatter is not supported with crs.Geodetic
        storm_rpts_plots[event_type] = storm_rpts_plot
        if drawrange > 0:
            logging.debug(f"about to draw tissot circles for {event_type}")
            # With lons and lats, specifying more than one dimension allows individual points to be drawn. 
            # Otherwise a grid of circles will be drawn.
            # It warns about using PlateCarree to approximate Geodetic. It still warps the circles
            # appropriately, so I think this is okay.
            within_range = ax.tissot(rad_km = drawrange.to("km").magnitude, lons=lons[np.newaxis], lats=lats[np.newaxis], 
                facecolor=kwdict[event_type]["c"], alpha=0.4, label=str(drawrange)+" range")
            # TODO: Legend does not support tissot cartopy.mpl.feature_artist.
            # A proxy artist may be used instead.
            # matplotlib.org/users/legend_guide.html#
            # creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists
            # storm_rpts_plots.append(within_range)

    return storm_rpts_plots



def to_MET(df, gribcode=None):
    # INPUT:
    #   df       - storm_reports DataFrame from spc.get_storm_reports()
    #   gribcode - Use gribcode if specified.
    # OUTPUT:
    #   txt - MET point observation format, one observation per line.
    #         encode event type (wind, hail, torn, significant or not) in "Grib_Code"
    #         assign EF-scale, wind speed, or hail size to "Observation_Value"
    # 
    # Each observation line will consist of the following 11 columns of data:
    met_columns = ["Message_Type", "Station_ID", "Valid_Time", "Lat", "Lon", "Elevation", "Grib_Code", "Level", "Height", "QC_String", "Observation_Value"]
    
    df["Message_Type"] = "StormReport" # as in https://github.com/dtcenter/METplus/blob/main_v3.1/parm/use_cases/model_applications/convection_allowing_models/read_ascii_storm.py
    df["Station_ID"]  = df.st
    df["Valid_Time"] = df.time.dt.strftime('%Y%m%d_%H%M%S') #df.time should be aware of UTC time zone
    df["Lat"]  = np.mean([df.slat,df.elat], axis=0)
    df["Lon"]  = circmean([df.slon,df.elon], low=-180, high=180, axis=0)
    df["Elevation"]  = 0.
    df["Height"] = 0.
    if gribcode:
        df["Grib_Code"] = gribcode
    else:
        df['Grib_Code'] = -9999
        # grib_code for tornado, hail and wind (regular and significant)
        # inspired by $MET_BASE/table_files/grib2_ndfd.txt
        # case-insensitive string matches
        df.loc[df['event_type'].str.lower() == 'torn', "Grib_Code"] = 197 # probability of tornado
        df.loc[df['event_type'].str.contains('hail',case=False), "Grib_Code"] = 198 # prob of hail
        df.loc[df['event_type'].str.contains('wind',case=False), "Grib_Code"] = 199 # prob of damaging wind
        df.loc[df['event_type'].str.lower() == 'large hail', "Grib_Code"] = 201 # prob of extreme hail
        df.loc[df['event_type'].str.lower() == 'high wind', "Grib_Code"] = 202 # prob of extreme wind
    df["Level"] = 0.
    df["QC_String"] = "NA" # DTC put 1,2,or 3 (torn, hail, or wind) in QC_String, but I like Grib_Code.
    df["Observation_Value"] = df.mag
    # index=False don't write index number
    # Change NaN to "NA" MET considers "NA" missing
    # This may not matter, but by adding a string 'NA', it changes the format of the entire column. Floats change to integers (or maybe strings). 
    df.replace(to_replace=np.nan, value = 'NA', inplace=True)
    txt = df.to_string(columns=met_columns,index=False,header=False)#used to append "\n" (for neatness?) but you can't split output on "\n" without empty string at end. 
    return txt


def listFD(url, ext=''):
    # Return list of files in url directory with given extension.
    from bs4 import BeautifulSoup
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]


# NCDC storm events homepage https://www.ncdc.noaa.gov/stormevents/ftp.jsp
#  event types: Hail, Thunderstorm Wind, Snow, Ice, etc. 
#               More types in https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/Storm-Data-Bulk-csv-Format.pdf
# There are 3 types of information: 1) details, 2) fatalities, and 3) locations
# NCDC storm event "details" don't have a lat and lon. Just a city, a range, and an azimuth.
# Perhaps the lat and lon are in the "location" files (as opposed to "details").
# Recommend using spc.get_storm_reports() instead.

def stormEvents(year, info="details", version="1.0"):
    # INPUT
    #   year   - 4-digit year
    # OUTPUT
    #   events - DataFrame with storm events
   
    # File name convention: StormEvents_details-ftp_v1.0_dyyyy_cyyyymmdd.csv
    # where dyyyy is the data year and cyyyymmdd is the creation date.

    url = f"https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles"
    for filename in listFD(url, "csv.gz"):
        if f"StormEvents_{info}-ftp_v{version}_d{year}_c" in filename:
            events = pd.read_csv(filename)
            break
    events["url"] = filename
    return events

def events2met(df):
    # Point Observation Format
    # input ASCII MET point observation format contains one observation per line. 
    # Each input observation line should consist of the following 11 columns of data:
    met_columns = ["Message_Type", "Station_ID", "Valid_Time", "Lat", "Lon", "Elevation", "Grib_Code", "Level", "Height", "QC_String", "Observation_Value"]
    
    df["Message_Type"] = "StormReport"
    df["Station_ID"]  = df.WFO
    df["Valid_Time"] = df.BEGIN.dt.strftime('%Y%m%d_%H%M%S')
    df["Lat"]  = df.lat # I don't know how to get this. 
    df["Lon"]  = df.lon
    df["Elevation"]  = "NA"
    df["Grib_Code"] = "NA"
    df["Level"] = "NA"
    df["Height"] = "NA"
    df["QC_String"] = "NA"
    df["Observation_Value"] = df.TOR_F_SCALE
    print(df.to_string(columns=met_columns))

