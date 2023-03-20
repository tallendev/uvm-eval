#!/bin/bash -xe

module load cuda mpi/openmpi-x86_64
for i in 0; do
export CUDA_VISIBLE_DEVICES=$i
./tealeaf
#cuda-memcheck ./tealeaf
done
