#!/bin/bash -l
#SBATCH --job-name=HWT_train
#SBATCH --account=NAML0001
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=256G
#SBATCH --output=train_mode_wwind.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cbecker@ucar.edu
module load gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
conda activate smode
python -u train_mode_cnn.py config/refl_32_patch.yml -t -i
