#!/bin/bash -l
#PBS -N HWT_post_proc
#PBS -A NAML0001
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o post_proc.out
#PBS -q casper
#PBS -l select=1:ncpus=12:mem=64GB
#PBS -m a
#PBS -M cbecker@ucar.edu
conda activate hwt
python -u /glade/work/cbecker/HWT_mode/post_process.py /glade/work/cbecker/HWT_mode/config/postproc_config.yml