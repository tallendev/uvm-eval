#!/bin/bash -xe

sudo rmmod -f nvidia-uvm                                                                                
sudo modprobe nvidia-uvm uvm_perf_prefetch_enable=0
module load cuda
export OMP_NUM_THREADS=1
nvprof ./build/bin/hpgmg-fv 7 7
