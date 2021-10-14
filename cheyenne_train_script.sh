#!/bin/bash -l
#PBS -N mode_cnn
#PBS -A NAML0001
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -q regular
#PBS -l select=1:ncpus=36
#PBS -l select=1:mem=109GB
#PBS -m abe
#PBS -M ggantos@ucar.edu
export PATH="/glade/u/home/ggantos/miniconda3/envs/mode_ch/bin:$PATH"
python -u train_mode_cnn.py config/ws_mode_cnn_train_201014_uh.yml -t -i -p
