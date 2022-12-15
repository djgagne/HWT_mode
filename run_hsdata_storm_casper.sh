#!/bin/bash -l
#SBATCH --job-name=hsdata_wrfrt
#SBATCH --account=NAML0001
#SBATCH --ntasks=25
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=dav
#SBATCH --mem=400G
#SBATCH --output=hsdata_wrfrt.%j.out
# Modify path to point to appropriate python environment
export PATH="/glade/u/home/dgagne/miniconda3/envs/goes/bin:$PATH"
#python -u /glade/u/home/dgagne/hagelslag/bin/hsdata /glade/u/home/dgagne/hagelslag/config/ncar_rt2020_data.config -p 20
python -u hsdata config/ncar_storm_data_3km_2020.config -p 25
