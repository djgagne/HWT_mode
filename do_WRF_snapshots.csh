#!/bin/csh

# Run WRF_snapshots.py for range of forecast hours and hourly offsets.


# 1st argument is "hyst" "ew" or "ws"
if ("$1" != "hyst" && "$1" != "ew" & "$1" != "ws") then
    echo 1st argument is "hyst" "ew" or "ws"
    exit 1
endif
set segmentation_method=$1
if ("$2" !~ 20????????) then
    echo 2nd argument is date yyyymmddhh
    exit 1
endif
set iyyyymmddhh=$2

set idir=/glade/scratch/ahijevyc
set tdir=$idir/track_data_ncarstorm_3km_REFL_COM_${segmentation_method}_csv
set pdir=$idir/track_data_ncarstorm_3km_REFL_COM_${segmentation_method}_nc
set odir=/glade/scratch/$USER/`basename $tdir`/$iyyyymmddhh # output to user's scratch directory
mkdir -pv $odir $odir/t2 $odir/capeshr

set iyyyy=`echo $iyyyymmddhh|cut -c1-4`
set imm=`echo $iyyyymmddhh|cut -c5-6`
set idd=`echo $iyyyymmddhh|cut -c7-8`
set ihh=`echo $iyyyymmddhh|cut -c9-10`
foreach t (-2 -1 0 1 2)
    # Forecast hours 14-34
    foreach fhr (`seq 14 34`) # expert labels start at fhr 16, not 14.
        # format date recognizes - it recognizes yyyymmdd but not yyyymmddhh.
        set iso="$iyyyy-$imm-${idd}T${ihh}:00+00:00"
        set valid=`date -u --date "$iso +${fhr}hours" +%Y%m%d%H`
        python WRF_snapshots.py $iyyyymmddhh $valid --no-counties --fill crefuh                                     --outdir $odir         --time $t --track $tdir --patch $pdir
        python WRF_snapshots.py $iyyyymmddhh $valid --no-counties --fill t2 --contour=cref_contour       --outdir $odir/t2      --time $t --track $tdir --patch $pdir
        python WRF_snapshots.py $iyyyymmddhh $valid --no-counties --fill mucape --barb shr06 --outdir $odir/capeshr --time $t --track $tdir --patch $pdir
    end
end

