#!/bin/bash -l
#PBS -N HWT_rt
#PBS -A NAML0001
#PBS -l walltime=00:20:00
#PBS -q casper
#PBS -l select=1:ncpus=12:mem=16GB
#PBS -m a
#PBS -M cbecker@ucar.edu
conda activate hagelslag
python -u /glade/work/cbecker/hagelslag/bin/hsdata /glade/work/cbecker/hagelslag/config/HRRR_AWS_Stream_real_time.config -j -p 12 2>&1 > "/glade/scratch/cbecker/HRRR_HWT_2022_real_time/logs/hagelslag_"$(date +"%Y-%m-%d_%H00")".log"

conda activate hwt
python -u /glade/work/cbecker/HWT_mode/run_mode_cnn.py /glade/work/cbecker/HWT_mode/config/HRRR_HWT_2022_rt.yml -e 2>&1 > "/glade/scratch/cbecker/HRRR_HWT_2022_real_time/logs/model_out_"$(date +"%Y-%m-%d_%H00")".log"