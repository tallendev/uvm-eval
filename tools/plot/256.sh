#!/bin/bash -xe

./batch_dist_plot.py ../../benchmarks/cublas/log_256/cublas_0.txt > 256r.txt
./batch_dist_plot.py ../../benchmarks/cublas/log_wq_256/cublas_0.txt > 256wq.txt
