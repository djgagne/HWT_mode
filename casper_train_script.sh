#!/bin/bash -l
#PBS -N mode_cnn
#PBS -A NAML0001
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -o train_mode.out
#PBS -q casper
#PBS -l select=1:ncpus=8:ngpus=1:mem=64GB -l gpu_type=v100

module load cuda/11 cudnn
conda activate hwt
python -u train_mode_cnn.py config/train_cnn_gmm.yml -t -i -u -p -p2