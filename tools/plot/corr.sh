#!/bin/bash -xe
#SBATCH -w n06
#SBATCH -J perf-corr
#SBATCH -t 16:00:00
#SBATCH --exclusive


#apps=( cublas)
#apps=( stream linear random cublas cufft) 
apps=(stream linear random cublas cufft hpgmg)
#apps=(hpgmg)
#apps=( stream linear random cublas cufft hpgmg tealeaf)

mkdir -p csvs
rm -f correlation.txt
for app in ${apps[@]}; do
    if [ ! -f "csvs/${app}.csv" ] || [ ! -f "csvs/${app}-pf.csv" ] || [ ! -f "csvs/${app}-all.csv" ]; then
        vmtouch -t ../../benchmarks/$app/log_*bsize_*/*_0.txt
        time python3 ./corr_csv.py -f ../../benchmarks/$app/log_*bsize_*/*_0.txt  -p ../../benchmarks/$app/quant-*.csv --app "$app"
    fi

    python3 ./corr.py ./csvs/${app}.csv -o "$app" >> correlation.txt
    python3 ./corr.py ./csvs/${app}-pf.csv -o "$app-pf" >> correlation-pf.txt
    python3 ./corr.py ./csvs/${app}-all.csv -o "$app-all" >> correlation-all.txt
done

#TODO try all
time python3 ./corr_csv.py -f ../../benchmarks/{stream,linear,random,cublas,hpgmg,cufft}/log_*bsize_*/*_0.txt  -p ../../benchmarks/{stream,linear,random,cublas,hpgmg,cufft}/quant-*.csv --app "all"
time python3 ./corr.py ./csvs/all.csv  -o "$app" >> correlation.txt
time python3 ./corr.py ./csvs/all-pf.csv  -o "all-pf" >> correlation-pf.txt
time python3 ./corr.py  ./csvs/all-all.csv  -o "all-all" >> correlation-all.txt
