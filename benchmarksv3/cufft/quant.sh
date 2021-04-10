#!/bin/bash -xe
#SBATCH -w voltron
#SBATCH -J cufft-quant
#SBATCH -t 8:00:00
#SBATCH --exclusive

module load cuda

cd /home/tnallen/cuda11/uvmmodel-NVIDIA-Linux-x86_64-450.51.05/kernel
make -j
sudo make modules_install
cd -

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

make clean
make

#size=`expr 80000000 / 6`
sizes=(20000000  40000000 60000000 80000000 100000000)
if [ "$#" -gt 0 ]; then
    sizes=($@)
fi

out="quant.csv"
rm -f $out
touch $out

name="cufft"
for i in ${sizes[@]}; do

    make clean
    make DEFS=-DSIGNAL_SIZE=$i
    echo "size,$i" >> $out 
    ./simpleCUFFT | grep "alloced,\|perf," >> $out

done

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
