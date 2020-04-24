#!/bin/bash -l
#PBS -N mode_cnn
#PBS -A NAML0001
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q regular
#PBS -l select=1:ncpus=36
#PBS -l select=mem=128GB
#PBS -m abe
#PBS -M dgagne@ucar.edu
export PATH="/glade/u/home/dgagne/miniconda3/envs/goes/bin:$PATH"
python -u train_mode_cnn.py config/ws_mode_cnn_train.yml -t -i -p
