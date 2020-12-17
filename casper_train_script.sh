#!/bin/bash -l
#SBATCH --job-name=HWT_train
#SBATCH --account=NAML0001
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=256G
#SBATCH --output=train_mode_cnn.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ggantos@ucar.edu
module load gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
ncar_pylib ncar_20200417
export PATH="/glade/work/ggantos/ncar_20200417/bin:$PATH"

pip install /glade/work/ggantos/HWT_mode/.
python -u train_mode_cnn.py config/ws_mode_cnn_train_201210_uh_masked.yml -t -i -p
