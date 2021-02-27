#!/bin/bash -l
#SBATCH --job-name=run_m_cnn
#SBATCH --account=NAML0001
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=dav
#SBATCH --mem=64G
#SBATCH --output=run_mode_cnn.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cbecker@ucar.edu
#SBATCH --gres=gpu:v100:1
module load cuda/10.1
conda activate hwt
#export PATH="/glade/u/home/dgagne/miniconda3/envs/goes/bin:$PATH"
#export PROJ_LIB="/glade/u/home/dgagne/miniconda3/envs/goes/share/proj"
python -u run_mode_cnn.py config/charlie_run.yml
