#!/bin/bash -l
#PBS -N mode_cnn
#PBS -A NAML0001
#PBS -l walltime=00:30:00
#PBS -o hwt_realtime.out
#PBS -e hwt_realtime.err
#PBS -q casper
#PBS -l select=1:ncpus=18:mem=64GB
#PBS -m a
#PBS -M cbecker@ucar.edu
conda activate hagelslag
python -u /glade/scratch/cbecker/hagelslag/bin/hsdata /glade/scratch/cbecker/hagelslag/config/HRRR_AWS_stream.config -p 12
conda activate hwt
python -u /glade/work/cbecker/HWT_mode/run_mode_cnn.py /glade/work/cbecker/HWT_mode/config/HRRR_run_cnn_gmm.yml