#!/bin/bash -xe
#SBATCH -w voltron
#SBATCH -J tealeaf-data
#SBATCH -t 8:00:00
#SBATCH --exclusive


cd /home/tnallen/cuda11/NVIDIA-Linux-x86_64-450.51.05/kernel
make -j
sudo make modules_install
cd -

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_batch_count=1

mkdir -p log2

./build.sh
name="tealeaf"

sizes=( small-tea.in big-tea.in smaller-tea.in )
for i in ${sizes[@]} ; do
    file=$i
    i=`basename -s .in $i`
    /home/tnallen/dev/uvm-learn/data/scripts/log2 /scratch1/"${name}_${i}" &

    pid=$!
    sleep 8
    cp $file tea.in


    ./tealeaf

    len=`cat /scratch1/"${name}_${i}" | wc -l`
    sleep 5
    until [ $(expr $(cat /scratch1/"${name}_${i}" | wc -l)  - ${len}) -eq 0 ]; do
        len=`cat /scratch1/"${name}_${i}" | wc -l`
        sleep 3
    done
    sleep 1
    kill $pid
done

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
