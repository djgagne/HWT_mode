#!/bin/bash -l
#SBATCH --job-name=run_m_cnn
#SBATCH --account=NAML0001
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=dav
#SBATCH --mem=128G
#SBATCH --output=run_mode_cnn.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dgagne@ucar.edu
#SBATCH --reservation=casper_8xV100
#SBATCH --gres=gpu:v100:1
module load cuda/10.1
export PATH="/glade/u/home/dgagne/miniconda3/envs/goes/bin:$PATH"
python -u run_mode_cnn.py config/ws_mode_cnn_run_20200504_uh.yml 
