#!/bin/bash -x
#SBATCH -w voltron
#SBATCH --exclusive
#SBATCH -J hpgmg-quant
#SBATCH -t 16:00:00
##SBATCH --mail-user=tnallen@clemson.edu
##SBATCH --mail-type=ALL

module load gcc cuda

./build.sh

# alternatively set CUDA_VISIBLE_DEVICES appropriately, see README for details
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_VISIBLE_DEVICES=0
CORES=32
MPI=1
# number of CPU threads executing coarse levels
export OMP_NUM_THREADS=$(($CORES/$MPI))
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
export OMP_PROC_BIND=true
export OMP_PLACES=threads
# enable threads for MVAPICH
#export MV2_ENABLE_AFFINITY=0

#SIZES=(7)
SIZES=(2 4 6 8 10 12 14 16 18 20)

right_now_string=`date +%Y_%m_%d__%H_%M_%S`
name="hpgmg"
#name="$name""$right_now_string"
ITER=1

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm 
out="quant2.csv"
echo "" > $out

for ((i=0; i < $ITER; i++)); do
    for s in ${SIZES[@]}; do
        echo "size,$s" >> $out
        time ./build/bin/hpgmg-fv  9 $s | grep "alloced,\|time," >> $out
        #srun  -n $MPI --cpu-bind=thread --cpus-per-task=$(($CORES/$MPI)) ./build/bin/hpgmg-fv $s 3 >> $out
    done
done

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm uvm_perf_prefetch_enable=0 

out="quant2-nopref.csv"
echo "" > $out

for ((i=0; i < $ITER; i++)); do
    for s in ${SIZES[@]}; do
        echo "size,$s" >> $out
        time ./build/bin/hpgmg-fv 7 $s | grep "alloced,\|time,"  >> $out
        #srun  -n $MPI --cpu-bind=thread --cpus-per-task=$(($CORES/$MPI)) ./build/bin/hpgmg-fv $s 3 >> $out
    done
done

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
