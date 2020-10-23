#!/bin/bash -l
#SBATCH --job-name=HWT_train
#SBATCH --account=NAML0001
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=256G
#SBATCH --output=train_mode_cnn.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ggantos@ucar.edu
module load cuda/10.1
export PATH="/glade/u/home/dgagne/miniconda3/envs/goes/bin:$PATH"
python -u train_mode_cnn.py config/ws_mode_cnn_train_201009_hm.yml -t -i -p
