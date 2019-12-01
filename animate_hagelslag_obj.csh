#!/bin/csh

# Make animated gifs of tracked objects 
# Run in directory with images created by WRF_snapshots.py.

set force=0 # clobber old gifs?
set debug=0
set p=""

while ("$1" != "")
	if ("$1" =~ d0[0-9]_*[0-9]) set p="$1"
	if ("$1" == "-f") set force=1
	if ("$1" == "-d") set debug=1
	shift
end

if ($debug) set echo

if ("$1" != "") then
	echo unknown argument $1
	exit
endif


if ("$p" != "") then
    if(-s $p.gif && ! $force)then
        echo $p.gif exists. Skipping.
        exit
    endif
    set h=`echo $p | cut -f2 -d- | cut -c6-7` 
    set h1=`echo $p | cut -f2 -d- | cut -c16-17`
    set fhr=`expr $h + $h1`
    set g="745x776+238+121"
    if($fhr<35)then 
        convert -crop $g +repage -loop 0 -delay 50 ${p}_-2.png ${p}_-1.png ${p}_+0.png ${p}_+1.png ${p}_+2.png ${p}_+2.png $p.gif
    else
        # forecast only goes to 36 hours. 
        convert -crop $g +repage -loop 0 -delay 50 ${p}_-2.png ${p}_-1.png ${p}_+0.png ${p}_+1.png ${p}_+1.png $p.gif
    endif
    if (-s $p.gif) echo created $p.gif
else
    set ts="" # Before Oct 20 2019, pngs don't
    set ts="_+0" # After Oct 20 2019, pngs have this time-shift string

    # Get all unique prefixes (different variables and dates)
    set prefixes=`ls d01*$ts.png|sed -e 's/\(.*\)\(_[0-9][0-9]_[0-9][0-9]_[0-9][0-9][0-9]_[0-9][0-9]\).*/\1/'|sort|uniq`
    echo $prefixes
    foreach prefix ($prefixes)
        
        echo prefix=$prefix
        echo "  count hrs"

        # number of a times each track age in hours is listed
        ls $prefix*$ts.png|sed -e 's/\(.*\)\([0-9][0-9]\).*/\2/'|sort|uniq -c
       
        # foreach track of a certain duration... 
        foreach n (`ls $prefix*$ts.png|sed -e 's/\(.*\)\([0-9][0-9]\).*/\2/'|sort|uniq`)
            if ($n < 3) continue
            echo ls $prefix\*${n}$ts.png
            foreach f (`ls $prefix*${n}$ts.png`)
                set id=`echo $f | sed -e "s/\(.*\)_\([0-9][0-9][0-9]\)_[0-9][0-9]$ts\.png/\2/"`
                echo $f $id
                set frames=$prefix*_${id}_*$ts.png
                set last_frame=`ls $frames|tail -n 1`
                set ofile=${prefix}_${id}$ts.gif
                if(-s $ofile && ! $force)then
                    echo $ofile exists. Skipping.
                else
                    convert -loop 500 -delay 30 $frames $last_frame $last_frame $ofile
                    echo made $n+1 frame $ofile
                endif
            end
        end
    end
endif
# endif needs a newline after it or else you get a "end: Not in while/foreach" error.
