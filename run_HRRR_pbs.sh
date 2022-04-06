#!/bin/bash -l
#PBS -N mode_cnn
#PBS -A NAML0001
#PBS -l walltime=02:00:00
#PBS -o hwt_hrrr.out
#PBS -e hwt_hrrr.out
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -m a
#PBS -M cbecker@ucar.edu
#conda activate hagelslag
#python -u /glade/work/cbecker/hagelslag/bin/hsdata /glade/work/cbecker/HWT_mode/config/HRRR_AWS.config -p 12
conda activate hwt
python -u /glade/work/cbecker/HWT_mode/run_mode_cnn.py /glade/work/cbecker/HWT_mode/config/HRRR_run_cnn_gmm.yml -e