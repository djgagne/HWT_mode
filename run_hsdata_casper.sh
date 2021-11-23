#!/bin/bash -l
#SBATCH --job-name=hsdata_wrfrt
#SBATCH --account=NAML0001
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=dav
#SBATCH --mem=128G
#SBATCH --output=hsdata_wrfrt.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cbecker@ucar.edu
#export PATH="/glade/u/home/dgagne/miniconda3/envs/goes/bin:$PATH"
conda activate hagelslag
python -u /glade/scratch/cbecker/hagelslag/bin/hsdata /glade/work/cbecker/HWT_mode/config/HRRR_hagelslag.config -p 24
