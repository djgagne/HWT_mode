import os # for NCARG_ROOT
import numpy as np
tenths = np.arange(0.1,1.1,0.1)
fifths = np.arange(0.2,1.2,0.2)

def readcm(name):
    '''Read colormap from file formatted as 0-1 RGB CSV'''
    projdir = '/glade/u/home/wrfrt/wwe/python_scripts/'
    fh = open(projdir+name, 'r')
    rgb = np.loadtxt(fh)
    fh.close()
    return rgb.tolist()

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib
       Replaces original function in /glade/u/home/wrfrt/wwe/python_scripts/fieldinfo.py
    '''

    # comments start with ; or #
    # first real line is ncolors = 256 (or something like that)
    # The rest is bunch of rgb values, one trio per line.

    fh = open(os.getenv('NCARG_ROOT','/glade/u/apps/opt/ncl/6.5.0/intel/17.0.1')+'/lib/ncarg/colormaps/%s.rgb'%name, 'r')
    rgb = np.loadtxt(fh, comments=[';', '#', 'n']) # treat ncolors=x as a comment
    fh.close()
    if rgb.max() > 1:
        rgb = rgb/255.0
    return rgb.tolist()


fieldinfo = {
  # surface and convection-related entries
  'precip'       :{ 'levels' : [0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.75,1,1.5,2,2.5,3.0], 'cmap': [readNCLcm('precip2_17lev')[i] for i in (0,1,2,4,5,6,7,8,10,12,13,14,15)], 'fname':['PREC_ACC_NC'] },
  'precip-24hr'  :{ 'levels' : [0,0.01,0.05,0.1,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12,13], 'cmap': [readNCLcm('precip2_17lev')[i] for i in (0,1,2,4,5,6,7,8,10,12,13,14,15,16,17)]+['#777777', '#AAAAAA', '#CCCCCC', '#EEEEEE']+[readNCLcm('sunshine_diff_12lev')[i] for i in (4,2,1)], 'fname':['PREC_ACC_NC'] },
  'precip-48hr'  :{ 'levels' : [0,0.05,0.1,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,12.0,15.0,20,25], 'cmap': [readNCLcm('precip2_17lev')[i] for i in (0,1,2,4,5,6,7,8,10,12,13,14,15,16,17)]+['#777777', '#AAAAAA', '#CCCCCC', '#EEEEEE']+[readNCLcm('sunshine_diff_12lev')[i] for i in (4,2,1)], 'fname':['PREC_ACC_NC'] },
  'precipacc':{ 'levels' : [0,0.01,0.05,0.1,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0], 'cmap': [readNCLcm('precip2_17lev')[i] for i in (0,1,2,4,5,6,7,8,10,12,13,14,15,16,17)]+['#777777', '#AAAAAA', '#CCCCCC', '#EEEEEE']+[readNCLcm('sunshine_diff_12lev')[i] for i in (4,2,1)], 'fname':['PREC_ACC_NC'] },
  'sbcape'       :{ 'levels' : [100,250,500,750,1000,1250,1500,1750,2000,2500,3000,3500,4000,4500,5000,5500,6000],
                    'cmap'   : ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname': ['SBCAPE'], 'filename':'upp' },
  'mlcape'       :{ 'levels' : [100,250,500,750,1000,1250,1500,1750,2000,2500,3000,3500,4000,4500,5000,5500,6000],
                    'cmap'   : ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname': ['MLCAPE'], 'filename':'upp' },
  'mucape'       :{ 'levels' : [10,25,50,100,250,500,750,1000,1250,1500,1750,2000,2500,3000,3500,4000,4500,5000,5500,6000],
                    'cmap'   : ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname': ['MUCAPE'], 'filename':'upp' },
  'sbcinh'       :{ 'levels' : [50,75,100,150,200,250,500], 'cmap': readNCLcm('topo_15lev')[1:], 'fname': ['SBCINH'], 'filename':'upp' },
  'mlcinh'       :{ 'levels' : [50,75,100,150,200,250,500], 'cmap': readNCLcm('topo_15lev')[1:], 'fname': ['MLCINH'], 'filename':'upp' },
  'pwat'         :{ 'levels' : [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0],
                    'cmap'   : ['#dddddd', '#cccccc', '#e1e1d3', '#e1d5b1', '#ffffd5', '#e5ffa7', '#addd8e', '#41ab5d', '#007837', '#005529', '#0029b1'],
                    'fname'  : ['PWAT'], 'filename':'upp'},
  'hailk1'       :{ 'levels' : [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0], 'cmap' : ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname':['HAIL_MAXK1'], 'filename': 'diag' },
  'hail2d'       :{ 'levels' : [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0], 'cmap' : ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname':['HAIL_MAX2D'], 'filename': 'diag' },
  'afhail'       :{ 'levels' : [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0], 'cmap' : ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname': ['AFWA_HAIL_NEWMEAN'], 'filename':'diag' },
  't2'           :{ 'levels' : [-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['T2'] },
  't2depart'     :{ 'levels' : [-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['T2'] },
  'mslp'         :{ 'levels' : [960,964,968,972,976,980,984,988,992,996,1000,1004,1008,1012,1016,1020,1024,1028,1032,1036,1040,1044,1048,1052], 'cmap':readNCLcm('nice_gfdl')[3:193], 'fname':['MSLP'], 'filename': 'upp' },
  'td2'          :{ 'levels' : [-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,64,68,72,76,80,84],
                    'cmap':['#ad598a', '#c589ac','#dcb8cd','#e7cfd1','#d0a0a4','#ad5960', '#8b131d', '#8b4513','#ad7c59', '#c5a289','#dcc7b8','#eeeeee', '#dddddd', '#bbbbbb', '#e1e1d3', '#e1d5b1','#ccb77a','#ffffe5','#f7fcb9', '#addd8e', '#41ab5d', '#006837', '#004529', '#195257', '#4c787c'],
                    'fname'  : ['DEWPOINT_2M'], 'filename':'upp'},
  'td2depart'    :{ 'levels' : [20,25,30,35,40,45,50,55,60,64,68,72,76,80,84], 'cmap'   : ['#eeeeee', '#dddddd', '#bbbbbb', '#e1e1d3', '#e1d5b1','#ccb77a','#ffffe5','#f7fcb9', '#addd8e', '#41ab5d', '#006837', '#004529', '#195257', '#4c787c'],
                    'fname'  : ['DEWPOINT_2M'], 'filename':'upp'},
  'thetapv'      :{  'levels' : np.arange(278,386,4), 'cmap': readNCLcm('WhiteBlueGreenYellowRed'), 'fname'  : ['theta_pv'], 'filename': 'diag'},
  'thetae'       :{  'levels' : [300,305,310,315,320,325,330,335,340,345,350,355,360], 'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname'  : ['T2', 'Q2', 'PSFC'], 'filename': 'diag'},
  'rh2m'           :{  'levels' : [0,10,20,30,40,50,60,70,80,90,100,110], 'cmap': readNCLcm('precip2_17lev')[:17][::-1], 'fname': ['T2', 'PSFC', 'Q2'], 'filename': 'diag'},
  'heatindex'    :{ 'levels' : [65,70,75,80,85,90,95,100,105,110,115,120,125,130], 'cmap': readNCLcm('MPL_hot')[::-1], 'fname': ['AFWA_HEATIDX'], 'filename':'diag' },
  'pblh'         :{ 'levels' : [0,250,500,750,1000,1250,1500,1750,2000,2500,3000,3500,4000],
                    'cmap': ['#eeeeee', '#dddddd', '#cccccc', '#bbbbbb', '#44aaee','#88bbff', '#aaccff', '#bbddff', '#efd6c1', '#e5c1a1', '#eebb32', '#bb9918'], 'fname': ['PBLH'] },
  'hmuh'         :{ 'levels' : [10,25,50,75,100,125,150,175,200,250,300,400,500], 'cmap': readNCLcm('prcp_1')[1:15], 'fname': ['UP_HELI_MAX'], 'filename':'diag'},
  'hmneguh'         :{ 'levels' : [10,25,50,75,100,125,150,175,200,250,300,400,500], 'cmap': readNCLcm('prcp_1')[1:15], 'fname': ['UP_HELI_MIN'], 'filename':'diag'},
  'hmuh03'       :{ 'levels' : [10,25,50,75,100,125,150,175,200,250,300,400,500], 'cmap': readNCLcm('prcp_1')[1:15], 'fname': ['UP_HELI_MAX03'], 'filename':'diag'},
  'hmuh01'       :{ 'levels' : [10,25,50,75,100,125,150,175,200,250,300,400,500], 'cmap': readNCLcm('prcp_1')[1:15], 'fname': ['UP_HELI_MAX01'], 'filename':'diag'},
  'rvort1'       :{ 'levels' : [0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015], 'cmap': readNCLcm('prcp_1')[1:15], 'fname': ['RVORT1_MAX'], 'filename':'diag'},
  'sspf'         :{ 'levels' : [10,25,50,75,100,125,150,175,200,250,300,400,500], 'cmap': readNCLcm('prcp_1')[:15], 'fname': ['updraft_helicity_max','WSPD10MAX','HAIL_MAXK1'], 'filename':'diag'},
  'hmup'         :{ 'levels' : [4,6,8,10,12,14,16,18,20,24,28,32,36,40,44,48], 'cmap': readNCLcm('prcp_1')[1:16], 'fname': ['W_UP_MAX'], 'filename':'diag' },
  'hmdn'         :{ 'levels' : [2,3,4,6,8,10,12,14,16,18,20,22,24,26,28,30], 'cmap': readNCLcm('prcp_1')[1:16], 'fname': ['W_DN_MAX'], 'filename':'diag' },
  'hmwind'       :{ 'levels' : [10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42], 'cmap': readNCLcm('prcp_1')[:16], 'fname': ['WSPD10MAX'], 'filename':'diag' },
  'hmgrp'        :{ 'levels' : [0.01,0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['GRPL_MAX'], 'filename':'diag' },
  'cref'         :{ 'levels' : [5,10,15,20,25,30,35,40,45,50,55,60,65,70], 'cmap': readcm('cmap_rad.rgb')[1:14], 'fname': ['REFL_MAX_COL'], 'filename':'upp' },
  'lmlref'       :{ 'levels' : [5,10,15,20,25,30,35,40,45,50,55,60,65,70], 'cmap': readcm('cmap_rad.rgb')[1:14], 'fname': ['REFL_10CM'], 'arraylevel':0 },
  'ref1km'       :{ 'levels' : [5,10,15,20,25,30,35,40,45,50,55,60,65,70], 'cmap': readcm('cmap_rad.rgb')[1:14], 'fname': ['REFL_1KM_AGL'], 'filename':'upp' },
  'echotop'      :{ 'levels' : [1000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000], 'cmap': readNCLcm('precip3_16lev')[1::], 'fname': ['ECHOTOP'], 'filename':'diag' },
  'srh3'         :{ 'levels' : [50,100,150,200,250,300,400,500], 'cmap': readNCLcm('perc2_9lev'), 'fname': ['SR_HELICITY_3KM'], 'filename' : 'upp' },
  'srh1'         :{ 'levels' : [50,100,150,200,250,300,400,500], 'cmap': readNCLcm('perc2_9lev'), 'fname': ['SR_HELICITY_1KM'], 'filename' : 'upp' },
  'shr06mag'     :{ 'levels' : [30,35,40,45,50,55,60,65,70,75,80], 'cmap': readNCLcm('perc2_9lev'), 'fname': ['UBSHR6', 'VBSHR6'], 'filename':'upp' },
  'shr01mag'     :{ 'levels' : [10,15,20,25,30,35,40,45,50,55], 'cmap': readNCLcm('perc2_9lev'), 'fname': ['UBSHR1', 'VBSHR1'], 'filename':'upp' },
  'zlfc'         :{ 'levels' : [0,250,500,750,1000,1250,1500,2000,2500,3000,3500,4000,5000], 'cmap': [readNCLcm('nice_gfdl')[i] for i in [3,20,37,54,72,89,106,123,141,158,175,193]], 'fname': ['AFWA_ZLFC'], 'filename':'diag' },
  'zlcl'         :{ 'levels' : [0,250,500,750,1000,1250,1500,2000,2500,3000,3500,4000,5000], 'cmap': [readNCLcm('nice_gfdl')[i] for i in [3,20,37,54,72,89,106,123,141,158,175,193]], 'fname': ['LCL_HEIGHT'], 'filename':'upp' },
  'ltg1'         :{ 'levels' : [0.1,0.5,1,1.5,2,2.5,3,4,5,6,7,8,10,12], 'cmap': readNCLcm('prcp_1')[:15], 'fname': ['LTG1_MAX'], 'filename':'diag' },
  'ltg2'         :{ 'levels' : [0.1,0.5,1,1.5,2,2.5,3,4,5,6,7,8,10,12], 'cmap': readNCLcm('prcp_1')[:15], 'fname': ['LTG2_MAX'], 'filename':'diag' },
  'ltg3'         :{ 'levels' : [0.1,0.5,1,1.5,2,2.5,3,4,5,6,7,8,10,12], 'cmap': readNCLcm('prcp_1')[:15], 'fname': ['LTG3_MAX'], 'filename':'diag' },
  'liftidx'      :{ 'levels' : [-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8], 'cmap': readNCLcm('nice_gfdl')[193:3:-1]+['#ffffff'], 'fname': ['SFC_LI'], 'filename':'upp'},
  'bmin'         :{ 'levels' : [-20,-16,-12,-10,-8,-6,-4,-2,-1,-0.5,0,0.5], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['MLBMIN'], 'filename':'upp' },
  'goesch3'      :{ 'levels' : [-80,-78,-76,-74,-72,-70,-68,-66,-64,-62,-60,-58,-56,-54,-52,-50,-48,-46,-44,-42,-40,-38,-36,-34,-32,-30,-28,-26,-24,-22,-20,-18,-16,-14,-12,-10], 'cmap': readcm('cmap_sat2.rgb')[38:1:-1], 'fname': ['GOESE_WV'], 'filename':'upp' },
  'goesch4'      :{ 'levels' : [-80,-76,-72,-68,-64,-60,-56,-52,-48,-44,-40,-36,-32,-28,-24,-20,-16,-12,-8,-4,0,4,8,12,16,20,24,28,32,36,40], 'cmap': readcm('cmap_satir.rgb')[32:1:-1], 'fname': ['GOESE_IR'], 'filename':'upp' },
  'afwavis'      :{ 'levels' : [0.0,0.1,0.25,0.5,1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0,12.0], 'cmap': readNCLcm('nice_gfdl')[3:175]+['#ffffff'], 'fname': ['VISIBILITY'], 'filename':'upp' },
  'pbmin'        :{ 'levels' : [0,30,60,90,120,150,180],'cmap': ['#dddddd', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1], 'fname': ['MLPBMIN','PBMIN_SFC'], 'filename':'upp' },
  'olrtoa'       :{ 'levels' : range(70,340,10), 'cmap': readcm('cmap_satir.rgb')[32:1:-1], 'fname': ['olrtoa'], 'filename':'diag' },

  # winter fields
  'snow'         :{ 'levels' : [0.01,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3,4,5], 'cmap':['#dddddd','#aaaaaa']+[readNCLcm('precip3_16lev')[i] for i in [1,3,5,8,10,12,14,16]]+['#FF99FF'], 'fname':['AFWA_SNOWFALL_HRLY'] }, # CSS  mod
  'snow-6hr'     :{ 'levels' : [0.25,0.5,0.75,1,2,3,4,5,6,8,10,12], 'cmap':['#dddddd','#aaaaaa']+[readNCLcm('precip3_16lev')[i] for i in [1,3,5,8,10,12,14,16]]+['#FF99FF'], 'fname':['AFWA_SNOWFALL_HRLY'] }, # CSS  mod
  'snow-12hr'    :{ 'levels' : [0.5,1,2,3,6,8,10,12,14,16,18,20], 'cmap':['#dddddd','#aaaaaa']+[readNCLcm('precip3_16lev')[i] for i in [1,3,5,8,10,12,14,16]]+['#FF99FF'], 'fname':['AFWA_SNOWFALL_HRLY'] }, # CSS  mod
  'snow-24hr'    :{ 'levels' : [1,3,6,8,10,12,15,18,21,24,30,36], 'cmap':['#dddddd','#aaaaaa']+[readNCLcm('precip3_16lev')[i] for i in [1,3,5,8,10,12,14,16]]+['#FF99FF'], 'fname':['AFWA_SNOWFALL_HRLY'] }, # CSS  mod
  'snowacc'      :{ 'levels' : [0.01,0.1,0.5,1,2,3,4,5,6,8,10,12,18,24,36,48,60], 'cmap':['#dddddd','#aaaaaa']+[readNCLcm('precip3_16lev')[i] for i in [1,2,3,4,5,6,8,10,11,12,13,15,16]]+['#FF99FF'], 'fname':['AFWA_SNOWFALL'], 'filename':'diag'}, # CSS mod colortable
  'iceacc'       :{ 'levels' : [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.25], 'cmap':[readNCLcm('precip3_16lev')[i] for i in [1,2,3,4,5,6,8,10,11,12,13,15,16]], 'fname':['AFWA_ICE'], 'filename':'diag'},
  'fzra'         :{ 'levels' : [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.25], 'cmap':[readNCLcm('precip3_16lev')[i] for i in [1,2,3,4,5,6,8,10,11,12,13,15,16]], 'fname':['AFWA_FZRA_HRLY'] }, # CSS added, hrly
  'fzraacc'      :{ 'levels' : [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.25], 'cmap':[readNCLcm('precip3_16lev')[i] for i in [1,2,3,4,5,6,8,10,11,12,13,15,16]], 'fname':['AFWA_FZRA'], 'filename':'diag'},
  'windchill'    :{ 'levels' : [-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45], 'cmap':readNCLcm('GMT_ocean')[20:], 'fname':['AFWA_WCHILL'], 'filename':'diag'},
  'freezelev'    :{ 'levels' : [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000], 'cmap':readNCLcm('nice_gfdl')[3:193], 'fname':['FZLEV'], 'filename':'diag'},
  'thck1000-500' :{ 'levels' : [480,486,492,498,504,510,516,522,528,534,540,546,552,558,564,570,576,582,588,592,600], 'cmap':readNCLcm('perc2_9lev'), 'fname':['GHT_PL', 'GHT_PL'], 'arraylevel':[0,5], 'filename':'diag'}, # CSS mod
  'thck1000-850' :{ 'levels' : [82,85,88,91,94,97,100,103,106,109,112,115,118,121,124,127,130,133,136,139,142,145,148,151,154,157,160], 'cmap':readNCLcm('perc2_9lev'), 'fname':['GHT_PL', 'GHT_PL'], 'arraylevel':[0,2], 'filename':'diag'}, # CSS mod

  # pressure level entries
  'hgt200'       :{ 'levels' : list(range(10900,12500,60)),                                                                                                            'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['GHT_PL'], 'filename':'diag', 'arraylevel':9},
  'hgt250'       :{ 'levels' : [9700,9760,9820,9880,9940,10000,10060,10120,10180,10240,10300,10360,10420,10480,10540,10600,10660,10720,10780,10840,10900,10960,11020], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['GHT_PL'], 'filename':'diag', 'arraylevel':8 },
  'hgt300'       :{ 'levels' : [8400,8460,8520,8580,8640,8700,8760,8820,8880,8940,9000,9060,9120,9180,9240,9300,9360,9420,9480,9540,9600,9660,9720,9780,9840,9900,9960,10020], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['GHT_PL'], 'filename':'diag', 'arraylevel':7 },
  'hgt500'       :{ 'levels' : [4800,4860,4920,4980,5040,5100,5160,5220,5280,5340,5400,5460,5520,5580,5640,5700,5760,5820,5880,5940,6000], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['GHT_PL'], 'filename':'diag', 'arraylevel':5 },
  'hgt700'       :{ 'levels' : [2700,2730,2760,2790,2820,2850,2880,2910,2940,2970,3000,3030,3060,3090,3120,3150,3180,3210,3240,3270,3300], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['GHT_PL'], 'filename':'diag', 'arraylevel':3 },
  'hgt850'       :{ 'levels' : [1200,1230,1260,1290,1320,1350,1380,1410,1440,1470,1500,1530,1560,1590,1620,1650], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['GHT_PL'], 'filename':'diag', 'arraylevel':2 },
  'hgt925'       :{ 'levels' : [550,580,610,640,670,700,730,760,790,820,850,880,910,940,970,1000,1030], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['GHT_PL'], 'filename':'diag', 'arraylevel':1 },
  'speed200'     :{ 'levels' : [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170], 'cmap': readNCLcm('wind_17lev'), 'fname': ['S_PL'], 'filename':'diag', 'arraylevel':9 },
  'speed250'     :{ 'levels' : [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170], 'cmap': readNCLcm('wind_17lev'), 'fname': ['S_PL'], 'filename':'diag', 'arraylevel':8 },
  'speed300'     :{ 'levels' : [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170], 'cmap': readNCLcm('wind_17lev'), 'fname': ['S_PL'], 'filename':'diag', 'arraylevel':7 },
  'speed500'     :{ 'levels' : [15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], 'cmap': readNCLcm('wind_17lev'), 'fname': ['S_PL'], 'filename':'diag', 'arraylevel':5 },
  'speed700'     :{ 'levels' : [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85], 'cmap': readNCLcm('wind_17lev'), 'fname': ['S_PL'], 'filename':'diag', 'arraylevel':3 },
  'speed850'     :{ 'levels' : [6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74], 'cmap': readNCLcm('wind_17lev'), 'fname': ['S_PL'], 'filename':'diag', 'arraylevel':2 },
  'speed925'     :{ 'levels' : [6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74], 'cmap': readNCLcm('wind_17lev'), 'fname': ['S_PL'], 'filename':'diag', 'arraylevel':1 },
  'temp200'      :{ 'levels' : [-65,-63,-61,-59,-57,-55,-53,-51,-49,-47,-45,-43,-41,-39,-37,-35,-33,-31,-29], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['T_PL'], 'filename':'diag', 'arraylevel':9 },
  'temp250'      :{ 'levels' : [-65,-63,-61,-59,-57,-55,-53,-51,-49,-47,-45,-43,-41,-39,-37,-35,-33,-31,-29], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['T_PL'], 'filename':'diag', 'arraylevel':8 },
  'temp300'      :{ 'levels' : [-65,-63,-61,-59,-57,-55,-53,-51,-49,-47,-45,-43,-41,-39,-37,-35,-33,-31,-29], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['T_PL'], 'filename':'diag', 'arraylevel':7 },
  'temp500'      :{ 'levels' : [-41,-39,-37,-35,-33,-31,-29,-26,-23,-20,-17,-14,-11,-8,-5,-2], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['T_PL'], 'filename':'diag', 'arraylevel':5 },
  'temp700'      :{ 'levels' : [-36,-33,-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['T_PL'], 'filename':'diag', 'arraylevel':3 },
  'temp850'      :{ 'levels' : [-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21,24,27,30], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['T_PL'], 'filename':'diag', 'arraylevel':2 },
  'temp925'      :{ 'levels' : [-24,-21,-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21,24,27,30,33], 'cmap': readNCLcm('nice_gfdl')[3:193], 'fname': ['T_PL'], 'filename':'diag', 'arraylevel':1 },
  'td500'        :{ 'levels' : [-30,-25,-20,-15,-10,-5,0,5,10], 'cmap' : readNCLcm('nice_gfdl')[3:193], 'fname': ['TD_PL'], 'filename':'diag', 'arraylevel':4 },
  'td700'        :{ 'levels' : [-30,-25,-20,-15,-10,-5,0,5,10], 'cmap' : readNCLcm('nice_gfdl')[3:193], 'fname': ['TD_PL'], 'filename':'diag', 'arraylevel':3 },
  'td850'        :{ 'levels' : [-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30], 'cmap' : readNCLcm('nice_gfdl')[3:193], 'fname': ['TD_PL'], 'filename':'diag', 'arraylevel':2 },
  'td925'        :{ 'levels' : [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30], 'cmap' : readNCLcm('nice_gfdl')[3:193], 'fname': ['TD_PL'], 'filename':'diag', 'arraylevel':1 },
  'rh300'        :{ 'levels' : [0,10,20,30,40,50,60,70,80,90,100], 'cmap' : [readNCLcm('MPL_PuOr')[i] for i in (2,18,34,50)]+[readNCLcm('MPL_Greens')[j] for j in (2,17,50,75,106,125)], 'fname': ['RH_PL'], 'filename':'diag', 'arraylevel':7 },
  'rh500'        :{ 'levels' : [0,10,20,30,40,50,60,70,80,90,100], 'cmap' : [readNCLcm('MPL_PuOr')[i] for i in (2,18,34,50)]+[readNCLcm('MPL_Greens')[j] for j in (2,17,50,75,106,125)], 'fname': ['RH_PL'], 'filename':'diag', 'arraylevel':5 },
  'rh700'        :{ 'levels' : [0,10,20,30,40,50,60,70,80,90,100], 'cmap' : readNCLcm('CBR_drywet'), 'fname': ['RH_PL'], 'filename':'diag', 'arraylevel':3 },
  'rh850'        :{ 'levels' : [0,10,20,30,40,50,60,70,80,90,100], 'cmap' : [readNCLcm('MPL_PuOr')[i] for i in (2,18,34,50)]+[readNCLcm('MPL_Greens')[j] for j in (2,17,50,75,106,125)], 'fname': ['RH_PL'], 'filename':'diag', 'arraylevel':2 },
  'rh925'        :{ 'levels' : [0,10,20,30,40,50,60,70,80,90,100], 'cmap' : [readNCLcm('MPL_PuOr')[i] for i in (2,18,34,50)]+[readNCLcm('MPL_Greens')[j] for j in (2,17,50,75,106,125)], 'fname': ['RH_PL'], 'filename':'diag', 'arraylevel':1 },
  'avo500'       :{ 'levels' : [0,9,12,15,18,21,24,27,30,33], 'cmap': readNCLcm('prcp_1'), 'fname': ['AVORT_PL'], 'filename':'diag', 'arraylevel':5 },
'pvort320k'    :{ 'levels' : [0,0.1,0.2,0.3,0.4,0.5,0.75,1,1.5,2,3,4,5,7,10],
                  'cmap'   : ['#ffffff','#eeeeee','#dddddd','#cccccc','#bbbbbb','#d1c5b1','#e1d5b9','#f1ead3','#003399','#0033FF','#0099FF','#00CCFF','#8866FF','#9933FF','#660099'],
                 'fname': ['PVORT_320K'], 'filename':'upp' },
 'bunkmag'      :{ 'levels' : [20,25,30,35,40,45,50,55,60], 'cmap':readNCLcm('wind_17lev')[1:], 'fname':['U_COMP_STM_6KM', 'V_COMP_STM_6KM'], 'filename':'upp' },
 'speed10m'     :{ 'levels' : [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51], 'cmap': readNCLcm('wind_17lev')[1:],'fname'  : ['U10', 'V10'], 'filename':'diag'},
 'speed10m-tc'  :{ 'levels' : [6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102] , 'cmap': readNCLcm('wind_17lev')[1:],'fname'  : ['U10', 'V10'], 'filename':'diag'},
 'stp'          :{ 'levels' : [0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0], 'cmap':readNCLcm('perc2_9lev'), 'fname':['SBCAPE','LCL_HEIGHT','SR_HELICITY_1KM','UBSHR6','VBSHR6'], 'arraylevel':[None,None,None,None,None], 'filename':'upp'},
 'uhratio'      :{ 'levels' : [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.2,1.3,1.4,1.5], 'cmap':readNCLcm('perc2_9lev'), 'fname':['updraft_helicity_max03', 'updraft_helicity_max'], 'filename':'diag'},
 'ptype'        :{ 'levels' : [0.01,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3,3.5,4], 'cmap':['#dddddd','#aaaaaa']+readNCLcm('precip3_16lev')[1:], 'fname':['AFWA_RAIN_HRLY', 'AFWA_FZRA_HRLY', 'AFWA_ICE_HRLY', 'AFWA_SNOWFALL_HRLY'], 'filename':'wrfout'},
 'winter'        :{ 'levels' : [0.01,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3,3.5,4], 'cmap':['#dddddd','#aaaaaa']+readNCLcm('precip3_16lev')[1:], 'fname':['AFWA_RAIN_HRLY', 'AFWA_FZRA_HRLY', 'AFWA_ICE_HRLY', 'AFWA_SNOWFALL_HRLY'], 'filename':'wrfout'},
  'crefuh'       :{ 'levels' : [5,10,15,20,25,30,35,40,45,50,55,60,65,70], 'cmap': readcm('cmap_rad.rgb')[0:13], 'fname': ['REFL_MAX_COL', 'MAX_UPDRAFT_HELICITY'], 'filename':'upp' },

  # wind barb entries
  'wind10m'      :{ 'fname'  : ['U10', 'V10'], 'filename':'diag', 'skip':40 },
  'wind250'      :{ 'fname'  : ['U_PL', 'V_PL'], 'filename':'diag', 'arraylevel':8, 'skip':40 },
  'wind300'      :{ 'fname'  : ['U_PL', 'V_PL'], 'filename':'diag', 'arraylevel':7, 'skip':40 },
  'wind500'      :{ 'fname'  : ['U_PL', 'V_PL'], 'filename':'diag', 'arraylevel':5, 'skip':40 },
  'wind700'      :{ 'fname'  : ['U_PL', 'V_PL'], 'filename':'diag', 'arraylevel':3, 'skip':40 },
  'wind850'      :{ 'fname'  : ['U_PL', 'V_PL'], 'filename':'diag', 'arraylevel':2, 'skip':40 },
  'wind925'      :{ 'fname'  : ['U_PL', 'V_PL'], 'filename':'diag', 'arraylevel':1, 'skip':40 },
  'shr06'        :{ 'fname'  : ['UBSHR6','VBSHR6'], 'filename': 'upp', 'skip':40 },
  'shr01'        :{ 'fname'  : ['UBSHR1', 'VBSHR1'], 'filename': 'upp', 'skip':40 },
  'bunkers'      :{ 'fname'  : ['U_COMP_STM_6KM', 'V_COMP_STM_6KM'], 'filename': 'upp', 'skip':40 },
}

# Copy fieldinfo dictionary for MPAS. Change some fnames and filenames.
mpas = fieldinfo
mpas['precip']['fname'] = ['rainnc']
mpas['precip-24hr']['fname'] = ['rainnc']
mpas['precip-48hr']['fname'] = ['rainnc']
mpas['precipacc']['fname'] = ['rainnc']
mpas['sbcape']['fname'] = ['sbcape']
mpas['sbcape']['filename'] = 'diag'
mpas['mlcape']['fname'] = ['mlcape']
mpas['mlcape']['filename'] = 'diag'
mpas['mucape']['fname'] = ['cape']
mpas['mucape']['filename'] = 'diag'
mpas['sbcinh']['fname'] = ['sbcin']
mpas['sbcinh']['filename'] = 'diag'
mpas['mlcinh']['fname'] = ['mlcin']
mpas['mlcinh']['filename'] = 'diag'
mpas['pwat']['fname'] = ['precipw']
mpas['pwat']['filename'] = 'diag'
mpas['mslp']['fname'] = ['mslp']
mpas['mslp']['filename'] = 'diag'
mpas['td2']['fname'] = ['surface_dewpoint']
mpas['td2depart']['fname'] = ['surface_dewpoint']
mpas['thetae']['fname'] = ['t2m', 'q2', 'surface_pressure']
mpas['rh2m']['fname'] = ['t2m', 'surface_pressure', 'q2']
mpas['pblh']['fname'] = ['hpbl']
mpas['hmuh']['fname'] = ['updraft_helicity_max']
mpas['hmuh03']['fname'] = ['updraft_helicity_max03']
mpas['hmuh01']['fname'] = ['updraft_helicity_max01']
mpas['rvort1']['fname'] = ['rvort1_max']
mpas['hmup']['fname'] = ['w_velocity_max']
mpas['hmdn']['fname'] = ['w_velocity_min']
mpas['hmwind']['fname'] = ['wind_speed_level1_max']
mpas['hmgrp']['fname'] = ['grpl_max']
mpas['cref']['fname'] = ['refl10cm_max']
mpas['cref']['filename'] = 'diag'
mpas['ref1km']['fname'] = ['refl10cm_1km']
mpas['ref1km']['filename'] = 'diag'
for ztop in ['3','1']:
    mpas['srh'+ztop]['fname'] = ['srh_0_'+ztop+'km']
    mpas['srh'+ztop]['filename'] =  'diag'
for ztop in ['6','1']:
    mpas['shr0'+ztop+'mag']['fname'] = ['uzonal_'+ztop+'km', 'umeridional_'+ztop+'km', 'uzonal_surface', 'umeridional_surface']
    mpas['shr0'+ztop+'mag']['filename'] = 'diag'
mpas['zlfc']['fname'] = ['lfc']
mpas['zlcl']['fname'] = ['lcl']
mpas['zlcl']['filename'] = 'diag' # only zlcl filename needed to be changed from upp, not zlfc
for plev in ['200', '250','300','500','700','850','925']:
    mpas['hgt'+plev]['fname'] = ['height_'+plev+'hPa']
    mpas['speed'+plev]['fname'] = ['uzonal_'+plev+'hPa','umeridional_'+plev+'hPa']
    del mpas['speed'+plev]['arraylevel']
    mpas['temp'+plev]['fname'] = ['temperature_'+plev+'hPa']
    del mpas['temp'+plev]['arraylevel']
for plev in ['500', '700', '850']:
    mpas['td'+plev]['fname'] = ['dewpoint_'+plev+'hPa']
    del mpas['td'+plev]['arraylevel']
    mpas['vort'+plev] = {'levels' : [0,9,12,15,18,21,24,27,30,33], 'cmap': readNCLcm('prcp_1'), 'fname': ['vorticity_'+plev+'hPa'], 'filename':'diag'}
mpas['vortpv']        = {'levels' : [0,9,12,15,18,21,24,27,30,33], 'cmap': readNCLcm('prcp_1'), 'fname': ['vort_pv'],               'filename':'diag'}
for plev in ['300', '500', '700', '850', '925']:
    mpas['rh'+plev]['fname'] = ['relhum_'+plev+'hPa']
mpas['speed10m']['fname'] = ['u10', 'v10']
mpas['speed10m-tc']['fname'] = ['u10','v10']
mpas['stp']['fname'] = ['sbcape','lcl','srh_0_1km','uzonal_6km','umeridional_6km','uzonal_surface','umeridional_surface']
mpas['stp']['filename'] = 'diag'
mpas['crefuh']['fname'] = ['refl10cm_max', 'updraft_helicity_max']
mpas['crefuh']['filename'] = 'diag'
mpas['wind10m']['fname'] = ['u10','v10']
mpas['shr06']  =  { 'fname'  : ['uzonal_6km','umeridional_6km','uzonal_surface','umeridional_surface'], 'filename': 'diag', 'skip':50 }
mpas['shr01']  =  { 'fname'  : ['uzonal_1km','umeridional_1km','uzonal_surface','umeridional_surface'], 'filename': 'diag', 'skip':50 }

# Enter wind barb info for list of pressure levels
for plev in ['200', '250', '300', '500', '700', '850', '925']:
    mpas['wind'+plev] = { 'fname' : ['uzonal_'+plev+'hPa', 'umeridional_'+plev+'hPa'], 'filename':'diag', 'skip':50}


# Another fieldinfo dictionary for nsc files. 
nsc = fieldinfo
nsc['crefuh']['fname'] = ['REFL_COM', 'UP_HELI_MAX']
nsc['crefuh']['threshold'] = 100
nsc['wind10m']['skip'] = 4
nsc['mucape']['fname'] = ['MUCAPE']
nsc['t2']['fname'] = ['T2']


# Combine levels from RAIN, FZRA, ICE, and SNOW for plotting 1-hr accumulated precip for each type. Ahijevych added this
#fieldinfo['ptypes']['levels'] = [fieldinfo['precip']['levels'][1:],fieldinfo['snow']['levels'],fieldinfo['ice']['levels'],fieldinfo['fzra']['levels']]

# domains = { 'domainname': { 'corners':[ll_lat,ll_lon,ur_lat,ur_lon], 'figsize':[w,h] } }
domains = { 'CONUS' :{ 'corners': [23.1593,-120.811,46.8857,-65.0212], 'fig_width': 1080 },
            'NA'  :{ 'corners': [15.00,-170.00,65.00,-50.00], 'fig_width':1080 },
            'SGP' :{ 'corners': [25.3,-107.00,36.00,-88.70], 'fig_width':1080 },
            'NGP' :{ 'corners': [40.00,-105.0,50.30,-82.00], 'fig_width':1080 },
            'CGP' :{ 'corners': [33.00,-107.50,45.00,-86.60], 'fig_width':1080 },
            'SW'  :{ 'corners': [28.00,-121.50,44.39,-102.10], 'fig_width':1080 },
            'NW'  :{ 'corners': [37.00,-124.40,51.60,-102.10], 'fig_width':1080 },
            'SE'  :{ 'corners': [26.10,-92.75,36.00,-71.00], 'fig_width':1080 },
            'NE'  :{ 'corners': [38.00,-91.00,46.80,-65.30], 'fig_width':1080 },
            'MATL':{ 'corners': [33.50,-92.25,41.50,-68.50], 'fig_width':1080 },
}
