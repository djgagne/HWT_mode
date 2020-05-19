#!/bin/bash -l
#SBATCH --job-name=real_m_cnn
#SBATCH --account=NAML0001
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=dav
#SBATCH --mem=128G
#SBATCH --output=real_m_cnn.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dgagne@ucar.edu
#SBATCH --reservation=casper_8xV100
module load cuda/10.1
export PATH="/glade/u/home/dgagne/miniconda3/envs/goes/bin:$PATH"
export PROJ_LIB="/glade/u/home/dgagne/miniconda3/envs/goes/share/proj"
python -u preprocess_ncar_wrf.py
python -u $HOME/hagelslag/bin/hsdata config/ncar_rt2020_data.config -p 1
python -u run_mode_cnn.py config/ws_mode_cnn_run_20200504_uh_real.yml 
