#!/bin/bash -xe

psize=$((4096 * 10))
#psize=$((4096 * 1))

echo "sequential kernel"
module load cuda
cd /home/tnallen/cuda11.2/NVIDIA-Linux-x86_64-460.27.04/kernel
make -j32
sudo make modules_install
cd -

sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
sudo rmmod -f nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

time ./matrixMul2 -wA=${psize} -hA=${psize} -wB=${psize} -hB=${psize}

echo "parallel kernel"
cd /home/tnallen/cuda11.2/wq-NVIDIA-Linux-x86_64-460.27.04/kernel
make
sudo make modules_install
cd -

sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
sudo rmmod -f nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

time ./matrixMul2 -wA=${psize} -hA=${psize} -wB=${psize} -hB=${psize}
