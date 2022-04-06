#!/bin/bash -l
#PBS -N mode_cnn
#PBS -A NAML0001
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o run_mode.out
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=64GB

module load cuda/11 cudnn
conda activate hwt
python -u run_mode_cnn.py config/WRF_run_cnn_gmm.yml