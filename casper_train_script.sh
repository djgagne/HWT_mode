#!/bin/bash -l
#SBATCH --job-name=train_m_cnn
#SBATCH --account=NAML0001
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=256G
#SBATCH --output=train_mode_cnn.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dgagne@ucar.edu
#SBATCH --reservation=casper_8xV100
module load cuda/10.1
export PATH="/glade/u/home/dgagne/miniconda3/envs/goes/bin:$PATH"
#python -u train_mode_cnn.py config/ws_mode_cnn_train_20200430.yml -t -i -p
python -u train_mode_cnn.py config/ws_mode_cnn_train_20200504_uh.yml -t -i -p
