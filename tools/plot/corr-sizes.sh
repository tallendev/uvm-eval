#!/bin/bash -xe
#SBATCH -w n06
#SBATCH -J perf-corr
#SBATCH -t 16:00:00
#SBATCH --exclusive


#apps=(cublas)
#apps=( stream linear random cublas cufft) 
apps=(stream linear random cublas cufft hpgmg)
#apps=(hpgmg)
#apps=( stream linear random cublas cufft hpgmg tealeaf)

mkdir -p csvs-size
rm -f correlation*sizes.txt

for app in ${apps[@]}; do
    vmtouch -t ../../benchmarks/$app/log_*bsize_*/*_0.txt
    time python3 ./corr_csv_sizes.py -f ../../benchmarks/$app/log_*bsize_*/*_0.txt  -p ../../benchmarks/$app/quant-*.csv --app "$app"

    python3 ./corr-sizes.py csvs-size/${app}*.csv >> correlation-sizes.txt

done
