#!/bin/bash -l
#PBS -N mode_cnn
#PBS -A NAML0001
#PBS -l walltime=04:00:00
#PBS -o hwt_WRF_UH.out
#PBS -e hwt_WRF_UH.out
#PBS -q casper
#PBS -l select=1:ncpus=12:mem=64GB
#PBS -m a
#PBS -M cbecker@ucar.edu
#conda activate hagelslag
#python -u /glade/scratch/cbecker/hagelslag/bin/hsdata /glade/scratch/cbecker/hagelslag/config/rt_WRF_2021.config -p 18
conda activate hwt
python -u /glade/work/cbecker/HWT_mode/run_mode_cnn.py /glade/work/cbecker/HWT_mode/config/WRF_run_cnn_gmm.yml -e