#!/bin/bash -xe
#SBATCH -N 1
#SBATCH -w n06
#SBATCH -J dist-plot
#SBATCH --exclusive
#SBATCH -t 48:00:00

bench_dir=../../benchmarks-drv2/

array=( ${bench_dir}/cublas/log_pf_16384_bsize_256/cublas_0.txt \
    ${bench_dir}/cublas/log_16384_bsize_256/cublas_0.txt \
    ${bench_dir}/stream/log_750004224_bsize_256/stream_0.txt \
    ${bench_dir}/stream/log_pf_750004224_bsize_256/stream_0.txt \
    ${bench_dir}/linear/log_8388608_bsize_256/linear_0.txt \
    ${bench_dir}/linear/log_pf_8388608_bsize_256/linear_0.txt \
    ${bench_dir}/random/log_8388608_bsize_256/random_0.txt \
    ${bench_dir}/random/log_pf_8388608_bsize_256/random_0.txt \
    ${bench_dir}/hpgmg/log_77_bsize_256/hpgmg_0.txt \
    ${bench_dir}/hpgmg/log_pf_77_bsize_256/hpgmg_0.txt \
    ${bench_dir}/hpgmg/log_88_bsize_256/hpgmg_0.txt \
    ${bench_dir}/hpgmg/log_pf_88_bsize_256/hpgmg_0.txt
    #${bench_dir}/cublas/log_32768_bsize_256/cublas_0.txt 
    #${bench_dir}/cublas/log_pf_32768_bsize_256/cublas_0.txt 
    )

#array=( ${bench_dir}/cublas/log_pf_16384_bsize_256/cublas_0.txt )

outdir="cublas-stuff"
mkdir -p $outdir
for a in ${array[@]}; do
    if ! [ -f $a ]; then
        echo "$a does not exist"
    else
        vmtouch -t $a
        fname=`dirname $a`
        fname=`basename $fname`
        #./dist_plot.py $a -o "${outdir}/${fname}-sm.png" > /dev/null &
        python3 batch_time.py $a &
        ./batch_dist_plot.py $a -d ./  > "${outdir}/${fname}.txt" &
    fi
    #./batch_dist_plot.py $a -d ${outdir} > "${outdir}/${fname}.txt" 
done

wait
