#!/bin/bash -ex
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J cufft-faults-quant
#SBATCH --exclusive
#SBATCH -t 48:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
ITERS=1

module load cuda

sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
sudo rmmod -f nvidia-uvm
sudo modprobe nvidia-uvm uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

cd ../../tools/syslogger/
make
cd -

bsizes=(256)
#bsizes=(32 64 128  256 512 1024 2048 4096 6144)
#bsizes=(32 64 128 256 512 1024 2048 4096 6144)

psizes=()
if [ $# -gt 0 ]; then
    for ((i=1; i<$#+1;i++));do
        psizes+=(${!i})
    done
else
    psizes=(80000000)
    #psizes=(80000000 16000000)
fi

cd ../cufft
make

pwd
rm -f quant*.csv
find . -maxdepth 1  -name 'log_*' -type d -exec rm -rf {} +

# collecting faults or quant data
cd ../../drivers/x86_64-460.27.04/batchd/kernel
    
make -j
sudo make modules_install
cd -
# prefetching on or off
#for ((pf=0; pf < 1; pf++)); do
for ((pf=0; pf < 2; pf++)); do
    # iter batch sizes
    for ((k=0; k<${#bsizes[@]}; k++)); do
        bsize=${bsizes[$k]}
        # iter problem size
        for ((i=0;i<${#psizes[@]}; i++)); do
            psize=${psizes[$i]}
            if [ $pf -eq 1 ]; then
                logdir="log_pf_${psize}_bsize_${bsizes[$k]}"
            else 
                logdir="log_${psize}_bsize_${bsizes[$k]}"
            fi
            mkdir -p $logdir

            # if want more than one sample per problem size
            for ((j=0;j<$ITERS;j++)); do
                sudo dmesg -C
                sudo rmmod -f nvidia-uvm
                sudo modprobe nvidia-uvm uvm_perf_prefetch_enable=${pf} uvm_perf_fault_batch_count=${bsizes[$k]}   #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
                
                if [ $pf -eq 1 ]; then
                    out="quant-pf-${bsize}.csv"
                else
                    out="quant-${bsize}.csv"
                fi

                file=cufft_$j
                logfile=/scratch1/$file
                pwd

                make clean
                make DEFS=-DSIGNAL_SIZE=$psize

                ../../tools/syslogger/log "$logfile" &
                pid=$!
                sleep 8

                echo "pid: $pid"
                #TODO idk if ar: actually works, don't think so

                make clean
                make DEFS=-DSIGNAL_SIZE=$psize
                echo "size,$psize" >> $out 
                ./simpleCUFFT | grep "alloced,\|perf," >> $out

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
    done
done

cd -




sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
