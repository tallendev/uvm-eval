#!/bin/bash -xe

#array=( ../../benchmarks/cublas/log_pf_16384/cublas_0.txt \
#    ../../benchmarks/cublas/log_pf_wq_16384/cublas_0.txt \ 
array=( ../../benchmarks/cublas/log_pf_32768/cublas_0.txt \
    ../../benchmarks/cublas/log_pf_wq_32768/cublas_0.txt \ 
    ../../benchmarks/hpgmg/log_pf_8/hpgmg_0.txt \ 
    ../../benchmarks/hpgmg/log_pf_wq_8/hpgmg_0.txt )

for a in ${array[@]}; do
    ./dist_plot.py $a
    ./batch_dist_plot.py $a
done
