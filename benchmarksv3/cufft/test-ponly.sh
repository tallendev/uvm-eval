#!/bin/bash -xe

module load cuda


echo "parallel kernel"
cd /home/tnallen/cuda11.2/wq-NVIDIA-Linux-x86_64-460.27.04/kernel
make -j32
sudo make modules_install
cd -

make
size=80000000
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
sudo rmmod  nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

for((i=0;i<1;i++)); do
    echo "i=$i"
    ./simpleCUFFT #$size
done
