#!/bin/bash -xe
module load julia



if ! [ -f sys_plots_`hostname`.so ]; then
    ./precompile.sh
fi

echo $outname

size="1.5"


bench=(../../benchmarks/hpgmg/log_pf_8/hpgmg_0.txt)
time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./batch_plot.jl -m ','  -n "hpgmg_88_256.png" ${bench[@]} -s $size &
#julia --compile=all -O3  ./pattern_plot.jl -m ','  -n "hpgmg_88_256.png" ${bench[@]} -s $size
time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./batch_plot.jl -m ','  -n "hpgmg_88_256_time.png" ${bench[@]} -s $size -t &

bench=(../../benchmarks/gauss-seidel/log_pf_22000000/gauss-seidel_0.txt)
#bench=(../../benchmarks/cublas/log_16384_bsize_256/cublas_0.txt)
time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./batch_plot.jl -m ','  -n "gauss-seidel_16384_256.png" ${bench[@]} -s $size  &
time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./batch_plot.jl -m ','  -n "gauss-seidel_16384_256_time.png" ${bench[@]} -s $size -t &



wait
