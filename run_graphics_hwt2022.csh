#!/bin/tcsh

#PBS -N hwt_graphics
#PBS -A NMMM0021
#PBS -l select=1:ncpus=11
#PBS -l walltime=00:30:00
#PBS -q casper
#PBS -j oe
#PBS -m e
#PBS -M sobash@ucar.edu

module load conda/latest
conda activate tf2py37

cd /glade/work/sobash/HWT_mode

echo 'plotting'
python plot_mode_predictions_realtime_hwt2022.py ${yyyymmddhh} obprobs
python plot_mode_predictions_realtime_hwt2022.py ${yyyymmddhh} nprobs

set numfiles = `ls /glade/scratch/sobash/realtime_graphics/${yyyymmddhh} | wc -l`
echo "${numfiles} graphics present on cheyenne" | mailx -s "cheyenne hwt graphics ${yyyymmddhh} -- ${numfiles}/2664 graphics" sobash@ucar.edu

echo 'rsyncing'
rsync -rtv --rsh=ssh /glade/scratch/sobash/realtime_graphics/${yyyymmddhh} sobash@whitedwarf.mmm.ucar.edu:/web/htdocs/projects/ncar_ensemble/hwtmode/graphics
