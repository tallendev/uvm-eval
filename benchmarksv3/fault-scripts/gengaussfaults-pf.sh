#!/bin/bash -ex
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J gauss-faults-pf
#SBATCH --exclusive
#SBATCH -t 48:00:00

ITERS=1

module load cuda
cd ../../drivers/faults-NVIDIA-Linux-x86_64-460.27.04/kernel
make
sudo make modules_install
cd -

sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
sudo rmmod -f nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

psizes=()
if [ $# -gt 0 ]; then
    for ((i=1; i<$#+1;i++));do
        psizes+=(${!i})
    done
else
    #psizes=(4096 $(expr 4096 \* 4) $(expr 4096 \* 8))
    psizes=(22000000)
fi

cd ../gauss-seidel
./build.sh
for ((i=0;i<${#psizes[@]}; i++)); do
    psize=${psizes[$i]}
    logdir="log_pf_${psize}"
    mkdir -p $logdir

    for ((j=0;j<$ITERS;j++)); do
        sudo dmesg -C
        sudo rmmod -f nvidia-uvm
        sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
        file=gauss-seidel_$j
        logfile=/scratch1/$file
        pwd

        /home/tnallen/dev/uvm-learn/data/scripts/log "$logfile" &
        pid=$!
        sleep 8

        echo "pid: $pid"
        time ./gs ${psize} #| grep "alloced,\|perf," >> $out

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
