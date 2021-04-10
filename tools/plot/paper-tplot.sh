#!/bin/bash -xe
module load julia



if ! [ -f sys_plots_`hostname`.so ]; then
    ./precompile.sh
fi

echo $outname

size=".5"

bench=(../../benchmarks/cublas/log_16384_bsize_256/cublas_0.txt)
time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./pattern_plot.jl -m ','  -n "cublas_16384_256.png" ${bench[@]} -s $size
time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./pattern_plot.jl -m ','  -n "cublas_16384_256_time.png" ${bench[@]} -s $size -t


wait
