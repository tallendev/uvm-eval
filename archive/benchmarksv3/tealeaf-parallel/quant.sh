#!/bin/bash -xe
#SBATCH -w voltron
#SBATCH -J tealeaf-quant
#SBATCH -t 8:00:00
#SBATCH --exclusive

cd /home/tnallen/cuda11/uvmmodel-NVIDIA-Linux-x86_64-450.51.05/kernel
make -j
sudo make modules_install
cd -


sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_batch_count=1

./build.sh
name="tealeaf"

out="quant.csv"
rm -f $out
touch $out

sizes=( small-tea.in big-tea.in smaller-tea.in )
for i in ${sizes[@]} ; do
    cp $i tea.in
    echo "size,$i" >> $out
    ./tealeaf | grep "alloced,\|perf," >> $out
done

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
