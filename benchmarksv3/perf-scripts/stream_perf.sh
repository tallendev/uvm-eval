#!/bin/bash -ex
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J stream-perf
#SBATCH --exclusive
#SBATCH -t 48:00:00

ITERS=5

module load cuda
cd /home/tnallen/cuda11/uvmmodel-NVIDIA-Linux-x86_64-450.51.05/kernel
make
sudo make modules_install
cd -

sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
sudo rmmod -f nvidia-uvm
sudo modprobe nvidia-uvm uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

psizes=()
if [ $# -gt 0 ]; then
    for ((i=1; i<$#+1;i++));do
        psizes+=(${!i})
    done
else
    psizes=(250001408 500002816) # 750004224)
fi

cd ../stream
for ((i=0;i<${#psizes[@]}; i++)); do
    psize=${psizes[$i]}

    make clean
    make

    logdir="plog_${psize}"
    mkdir -p $logdir

    file=stream.perf
    logfile=$logdir/$file
    rm -f $logfile
    echo "writing log to \"`pwd`/$logfile\""
    for ((j=0;j<$ITERS;j++)); do
        ./cuda-stream -n 1 -s $psize --triad-only |& grep "Bandwidth" | awk '{print $3}' &>> $logfile
    done

done
cd -

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
