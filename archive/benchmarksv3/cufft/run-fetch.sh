#!/bin/bash -xe
#SBATCH -w voltron
#SBATCH -J cufft-data
#SBATCH -t 8:00:00
#SBATCH --exclusive

module load cuda

cd /home/tnallen/cuda11/NVIDIA-Linux-x86_64-450.51.05/kernel
make -j
sudo make modules_install
cd -

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

logdir=log-fetch
mkdir -p $logdir
make clean
make

#size=`expr 80000000 / 6`
size=80000000
if [ "$#" -eq 1 ]; then
    size=$1
fi

name="cufft"
i=$size

/home/tnallen/dev/uvm-learn/data/scripts/log /scratch1/"${name}_${i}" &

pid=$!
sleep 8

./simpleCUFFT #$size

len=`cat /scratch1/"${name}_${i}" | wc -l`
sleep 5
until [ $(expr $(cat /scratch1/"${name}_${i}" | wc -l)  - ${len}) -eq 0 ]; do
    len=`cat /scratch1/"${name}_${i}" | wc -l`
    sleep 3
done
sleep 1
kill $pid


sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
