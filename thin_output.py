#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys, os

yyyymmdd = sys.argv[1]

# retain mean/max/min for each field
usecols = 'Step_ID,UP_HELI_MAX_max,Centroid_Lat,Centroid_Lon' 

fname = '/glade/scratch/ahijevyc/track_data_ncarstorm_3km_REFL_COM_hyst_csv/track_step_NCARSTORM_d01_%s-0000.csv'%yyyymmdd
if os.path.exists(fname):
        print 'thinning', fname
        df = pd.read_csv(fname, usecols=usecols.split(','))

        df.to_csv('test.csv', float_format='%.3f', header=False, index=False, columns=usecols.split(','))
