#!/bin/bash -xe

psize=$((4096 * 256))
#psize=$((4096 * 1))


sudo dmesg -C
cd /home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel
make -j32
sudo make modules_install
cd -

sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
sudo rmmod  nvidia-uvm
sudo modprobe nvidia-uvm uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

make DEFS=-DPNUM=${psize}

./page
