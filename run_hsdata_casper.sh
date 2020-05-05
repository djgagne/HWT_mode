#!/bin/bash -l
#SBATCH --job-name=hsdata_wrfrt
#SBATCH --account=NAML0001
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --partition=dav
#SBATCH --mem=384G
#SBATCH --output=hsdata_wrfrt.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dgagne@ucar.edu
export PATH="/glade/u/home/dgagne/miniconda3/envs/goes/bin:$PATH"
hsdata /glade/u/home/dgagne/hagelslag/config/ncar_rt2020_data.config -p 12
