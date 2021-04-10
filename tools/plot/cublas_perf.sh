#!/bin/bash -xe

shopt -s extglob

#ls ../../benchmarks/cublas/quant-*.csv
#python3 ./perf_bs_plot.py ../../benchmarks/cublas/quant-[0-9]*.csv -o "cublas-perf.png"
#python3 ./perf_bs_plot.py ../../benchmarks/cublas/quant-pf-*.csv -o "cublas-pf-perf.png"


python3 ./perf_bs_plot.py ../../benchmarks/stream/quant-[0-9]*.csv -o "stream-perf.png"
python3 ./perf_bs_plot.py ../../benchmarks/stream/quant-pf-*.csv -o "stream-pf-perf.png"


python3 ./perf_bs_plot.py ../../benchmarks/linear/quant-[0-9]*.csv -o "linear-perf.png"
python3 ./perf_bs_plot.py ../../benchmarks/linear/quant-pf-*.csv -o "linear-pf-perf.png"
