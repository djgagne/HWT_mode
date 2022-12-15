#!/bin/bash -l
source ~/.bashrc

cd /glade/scratch/cbecker/HRRR_HWT_2022_real_time/model_output/evaluation/
ls -1 > ../available_dates.csv

aws s3 sync --acl public-read /glade/scratch/cbecker/HRRR_HWT_2022_real_time s3://storm-mode >> /glade/scratch/cbecker/HRRR_HWT_2022_real_time/cron.log
