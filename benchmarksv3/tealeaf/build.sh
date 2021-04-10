#!/bin/bash
make clean
module load mpi/openmpi-x86_64 cuda
make -j DEBUG=yes
