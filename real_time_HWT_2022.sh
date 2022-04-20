#!/bin/bash -l
#PBS -N HWT_rt
#PBS -A NAML0001
#PBS -l walltime=00:20:00
#PBS -o hwt_rt_2022.out
#PBS -e hwt_rt_2022.out
#PBS -q casper
#PBS -l select=1:ncpus=12:mem=16GB
#PBS -m a
#PBS -M cbecker@ucar.edu
conda activate hagelslag
python -u /glade/work/cbecker/hagelslag/bin/hsdata /glade/work/cbecker/hagelslag/config/HRRR_AWS_Stream.config -j -p 12

conda activate hwt
python -u /glade/work/cbecker/HWT_mode/run_mode_cnn.py /glade/work/cbecker/HWT_mode/config/HRRR_run_cnn_gmm.yml -e