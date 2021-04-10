#!/bin/bash -xe
#SBATCH -w n06
#SBATCH -J stats-plot
#SBATCH -t 16:00:00
#SBATCH --exclusive


#apps=( cublas)
#apps=( stream linear random cublas cufft) 
apps=(stream linear random cublas cufft hpgmg)
#apps=(hpgmg)
#apps=( stream linear random cublas cufft hpgmg tealeaf)

for app in ${apps[@]}; do
    python3 ./stats-plot.py csvs/$app.csv
    python3 ./stats-plot.py csvs/$app-pf.csv

done

