#! /bin/bash -x
#SBATCH -w voltron
#SBATCH -J tealeaf-sgemm-data
#SBATCH -t 8:00:00
##SBATCH --mail-user=tnallen@clemson.edu
##SBATCH --mail-type=ALL


module load cuda



sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=1 uvm_perf_fault_batch_count=1

# clear dmesg
right_now_string=`date +%Y_%m_%d__%H_%M_%S`
name="tealeaf"
name="$name" #"$right_now_string"
make


mkdir -p log


#sizes=( small-tea big-tea smaller-tea.)
sizes=( smaller-tea small-tea ) #big-tea )

for i in ${sizes[@]} ; do
    file=$i.in
    /home/tnallen/dev/uvm-learn/data/scripts/log /scratch1/"${name}_${i}" &

    pid=$!
    sleep 8
    cp $file tea.in

    ./run.sh $i

    len=`cat /scratch1/"${name}_${i}" | wc -l`
    sleep 5
    until [ $(expr $(cat /scratch1/"${name}_${i}" | wc -l)  - ${len}) -eq 0 ]; do
        len=`cat /scratch1/"${name}_${i}" | wc -l`
        sleep 3
    done
    sleep 1
    kill $pid

done
ls ./log/

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

