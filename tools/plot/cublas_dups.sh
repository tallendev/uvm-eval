#!/bin/bash -xe

shopt -s extglob

#ls ../../benchmarks/cublas/quant-*.csv
#python3 ./perf_bs_plot.py ../../benchmarks/cublas/quant-[0-9]*.csv -o "cublas-perf.png"
#python3 ./perf_bs_plot.py ../../benchmarks/cublas/quant-pf-*.csv -o "cublas-pf-perf.png"


python3 ./duplicates_plot.py ../../benchmarks/stream/log_[0-9]*_bsize_*/*_0.txt -o "stream-dups.png"
python3 ./duplicates_plot.py ../../benchmarks/stream/log_pf_[0-9]*_bsize_*/*_0.txt -o "stream-pf-dups.png"

python3 ./duplicates_plot.py ../../benchmarks/linear/log_pf_[0-9]*_bsize_*/*_0.txt -o "linear-pf-dups.png"
python3 ./duplicates_plot.py ../../benchmarks/linear/log_[0-9]*_bsize_*/*_0.txt -o "linear-dups.png"
