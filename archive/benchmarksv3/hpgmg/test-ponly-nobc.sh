#!/bin/bash -xe

size=(8 8)
#psize=$((4096 * 1))


echo "parallel kernel"
cd /home/tnallen/cuda11.2/wq-nobc-NVIDIA-Linux-x86_64-460.27.04/kernel
make -j32
sudo make modules_install
cd -

module load cuda mpi/openmpi-x86_64

sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
sudo rmmod  nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

for((i=0;i<1;i++)); do
    echo "i=$i"
    time ./build/bin/hpgmg-fv ${size[@]}
done
