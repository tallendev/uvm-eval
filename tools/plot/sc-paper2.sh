#!/bin/bash -xe
module load julia



if ! [ -f sys_plots_`hostname`.so ]; then
    ./precompile.sh
fi

echo $outname

size=".5"


bench=(../../benchmarks/hpgmg/log_pf_8/hpgmg_0.txt)
time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./prefetch_plot.jl -m ','  -n "hpgmg_88_256.png" ${bench[@]} -s $size -f

bench=(../../benchmarks/gauss-seidel/log_pf_22000000/gauss-seidel_0.txt)
time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./prefetch_plot.jl -m ','  -n "gauss-seidel_16384_256.png" ${bench[@]} -s $size -f



wait
