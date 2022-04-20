#!/bin/bash -l
#PBS -N rt_mode
#PBS -A NAML0001
#PBS -l walltime=00:30:00
#PBS -o hwt_rt.out
#PBS -e hwt_rt.out
#PBS -q casper
#PBS -l select=1:ncpus=12:mem=64GB
#PBS -m a
#PBS -M cbecker@ucar.edu
cd /glade/scratch/cbecker/HWT_2022_rt_test/
conda activate hagelslag
python -u /glade/work/cbecker/hagelslag/bin/hsdata /glade/work/cbecker/HWT_mode/config/HWT_2022_rt_hrrr_hs_config.py -p 12
conda activate hwt
python -u /glade/work/cbecker/HWT_mode/run_mode_cnn.py /glade/work/cbecker/HWT_mode/config/HWT_2022_rt_hrrr_run.yml -e