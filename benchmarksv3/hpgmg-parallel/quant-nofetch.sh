#!/bin/bash -ex
#SBATCH -w voltron
#SBATCH --exclusive
#SBATCH -J hpgmg-quant-nofetch
#SBATCH -t 16:00:00
##SBATCH --mail-user=tnallen@clemson.edu
##SBATCH --mail-type=ALL

module load cuda

cd /home/tnallen/cuda11/uvmmodel-NVIDIA-Linux-x86_64-450.51.05/kernel
make
sudo make modules_install
cd -

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

oldifs=$IFS
export IFS=""
SIZES=("6 7" "7 7" "7 8" "8 8")

name="hpgmg"
ITER=1

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm uvm_perf_prefetch_enable=0 

out="quant-nofetch.csv"
rm -f $out
touch $out

for ((i=0; i < $ITER; i++)); do
    for s in ${SIZES[@]}; do
        echo "size,$s" >> $out
        val=$s
        export IFS=$oldifs
        time ./build/bin/hpgmg-fv $s | grep "alloced,\|perf,"  >> $out
        export IFS=""
        #srun  -n $MPI --cpu-bind=thread --cpus-per-task=$(($CORES/$MPI)) ./build/bin/hpgmg-fv $s 3 >> $out
    done
done

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
