#!/bin/bash -ex
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J abcd-faults
#SBATCH --exclusive
#SBATCH -t 48:00:00

ITERS=5

module load cuda
cd /home/tnallen/cuda11/NVIDIA-Linux-x86_64-450.51.05/kernel
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
    psizes=(32)
fi

cd ../abcd
for ((i=0;i<${#psizes[@]}; i++)); do
    psize=${psizes[$i]}

    make clean
    make DEFS=-DTSIZE=$psize

    logdir="log_${psize}_k"
    mkdir -p $logdir

    for ((j=0;j<$ITERS;j++)); do
        file=abcd_$j
        logfile=/scratch1/$file
        pwd

        /home/tnallen/dev/uvm-learn/data/scripts/log "$logfile" &
        pid=$!
        sleep 8

        echo "pid: $pid"
        time ./abcd

        len=`cat "$logfile" | wc -l`
        sleep 5
        until [ $(expr $(cat "$logfile" | wc -l)  - ${len}) -eq 0 ]; do
            len=`cat "$logfile" | wc -l`
            sleep 3
        done
        sleep 1
        kill $pid
        mv $logfile $logdir/
        ../../tools/sys2csv/log2csv.sh $logdir/$file
    done

done
cd -

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
