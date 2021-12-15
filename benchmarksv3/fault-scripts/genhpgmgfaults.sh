#!/bin/bash -ex
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J hpgmg-faults
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

oldifs=$IFS
export IFS=""
psizes=()
if [ $# -gt 0 ]; then
    for ((i=1; i<$#+1;i++));do
        psizes+=(${!i})
    done
else
    psizes=(8 8)
    #psizes=("6 7" "7 7" "7 8" "8 8" "9 9")
fi


cd ../hpgmg
if [ -d "build" ]; then
    make clean
fi
rm -rf build
./build.sh

for ((i=0;i<${#psizes[@]}; i++)); do
    sudo dmesg -C
    psize=${psizes[$i]}
    logdir="log_${psize}"
    mkdir -p $logdir

    for ((j=0;j<$ITERS;j++)); do
        file=hpgmg_$j
        logfile=/scratch1/$file
        pwd

        ../../tools/syslogger/log "$logfile" &
        pid=$!
        sleep 8

        echo "pid: $pid"
        export IFS=$oldifs
        time ./build/bin/hpgmg-fv ${psizes[@]}
        export IFS=""

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
