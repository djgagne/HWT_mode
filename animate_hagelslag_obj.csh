#!/bin/csh

# Make animated gifs of tracked objects 
# Run in directory with images created by WRF_snapshots.py.

set force=0 # clobber old gifs?

set ts="_+0" # After Oct 20 2019, pngs have this time-shift string
set ts="" # Before Oct 20 2019, pngs don't

# Get all unique prefixes (different variables and dates)
set prefixes=`ls d01*$ts.png|cut -d_ -f 1-5|sort|uniq`
foreach prefix ($prefixes)
    
    # works for REFL_1KM_AGL_ because -f 9 accounts for extra underscores
    # works for W_UP_MAX_ because -f 9 accounts for extra underscores
    if ($prefix =~ *REFL_1KM_AGL_*) set w=9
    if ($prefix =~ *W_UP_MAX_*) set w=9
    if ($prefix =~ *REFL_COM_*) set w=8
    set w_1=`expr $w - 1`

    echo prefix=$prefix
    echo "  count hrs"
    ls $prefix*$ts.png|cut -d_ -f $w|cut -c1-2|sort|uniq -c

    foreach n (`ls $prefix*$ts.png|cut -d_ -f $w|cut -c1-2|sort|uniq`)
        if ($n < 3) continue
        foreach f (`ls $prefix*${n}$ts.png`)
            set id=`echo $f | cut -d_ -f $w_1`
            set frames=$prefix*_${id}_*$ts.png
            set last_frame=`ls $frames|tail -n 1`
            set ofile=${prefix}_${id}$ts.gif
            if(! -s $ofile || $force)then
                convert -loop 500 -delay 30 $frames $last_frame $last_frame $ofile
                echo made $n+1 frame $ofile
            endif
        end
    end
end
