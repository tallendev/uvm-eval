#!/bin/bash -xe
#SBATCH -w n06
#SBATCH -J perf-plots
#SBATCH -t 16:00:00
#SBATCH --exclusive

shopt -s extglob

#apps=( cublas stream linear )
#apps=( stream linear random cublas cufft)
#apps=( stream linear random cublas cufft hpgmg)
apps=( stream linear random cublas cufft tealeaf hpgmg)
#apps=(  hpgmg)
#apps=( tealeaf hpgmg)
#apps=(tealeaf)
#apps=(tealeaf)
#apps=( stream tealeaf hpgmg)

#ls ../../benchmarks/cublas/quant-*.csv
#python3 ./perf_bs_plot.py ../../benchmarks/cublas/quant-[0-9]*.csv -o "cublas-perf.png"
#python3 ./perf_bs_plot.py ../../benchmarks/cublas/quant-pf-*.csv -o "cublas-pf-perf.png"
date
for app in ${apps[@]}; do

    vmtouch -t ../../benchmarks/$app/quant-[^pf]*.csv ../../benchmarks/$app/log_[^pf]*_bsize_*/*_0.txt ../../benchmarks/$app/quant-pf*.csv ../../benchmarks/$app/log_pf_*_bsize_*/*_0.txt
    python3 ./perf_bs_plot.py ../../benchmarks/$app/quant-[^pf]*.csv -o "$app-perf.png" &
    python3 ./perf_bs_plot.py ../../benchmarks/$app/quant-pf-*.csv -o "$app-pf-perf.png" &
    python3 ./perf_stats.py ../../benchmarks/$app/log_[^pf]*_bsize_*/*_0.txt -o "$app" &
    python3 ./perf_stats.py ../../benchmarks/$app/log_pf_*_bsize_*/*_0.txt -o "$app-pf" &

    wait

done
date
echo "done"
