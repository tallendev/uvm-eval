#!/bin/bash -ex
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J stream-faults
#SBATCH --exclusive
#SBATCH -t 48:00:00

export IGNORE_CC_MISMATCH=1
ITERS=1

module load cuda
cd ../../drivers/faults-NVIDIA-Linux-x86_64-460.27.04/kernel
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
    psizes=(250001408 500002816 750004224)
    #psizes=( 256 )
fi

cd ../stream
make clean
make
for ((i=0;i<${#psizes[@]}; i++)); do
    psize=${psizes[$i]}
    logdir="log_${psize}"
    mkdir -p $logdir

    for ((j=0;j<$ITERS;j++)); do
        sudo dmesg -C
        sudo rmmod -f nvidia-uvm
        sudo modprobe nvidia-uvm uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
        file=stream_$j
        logfile=/scratch1/$file
        pwd

        ../../tools/syslogger/log "$logfile" &
        pid=$!
        sleep 8

        echo "pid: $pid"
        time ./cuda-stream -n 1 -s $psize --triad-only

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
        dmesg > $logdir/dmesg_$i
    done

done
cd -

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1