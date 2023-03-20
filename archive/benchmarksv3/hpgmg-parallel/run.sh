#!/bin/bash -xe
#SBATCH -w voltron
#SBATCH -J hpgmg-data
#SBATCH -t 48:00:00
##SBATCH --mail-user=tnallen@clemson.edu
##SBATCH --mail-type=ALL

module load cuda mpi

cd /home/tnallen/cuda11/NVIDIA-Linux-x86_64-450.51.05/kernel
make
sudo make modules_install
cd -

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=1 uvm_perf_fault_batch_count=1

name="hpgmg"
ITER=1


# alternatively set CUDA_VISIBLE_DEVICES appropriately, see README for details
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
CORES=32
MPI=1
# number of CPU threads executing coarse levels
export OMP_NUM_THREADS=$(($CORES/$MPI))
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
export OMP_PROC_BIND=true
export OMP_PLACES=threads
# enable threads for MVAPICH
#export MV2_ENABLE_AFFINITY=0

echo "#@$%@#%JIO#P$%JIOLGVBFIOJGESRIOJGEOIRJSIOJLGERSOIJL"
echo $@
if [ "$#" -gt 0 ] ; then
    size=( $1 $2 )
else
    size=( 7 7)
fi    


mkdir -p log
for ((i=0; i < $ITER; i++)); do
    suffix=`echo ${size[@]} | sed "s/ /_/g"`
    iname=${name}_${suffix}
    /home/tnallen/dev/uvm-learn/data/scripts/log /scratch1/"${iname}" &

    pid=$!
    sleep 8

    ./build/bin/hpgmg-fv ${size[@]}

    len=`cat /scratch1/"${iname}" | wc -l`
    sleep 5
    until [ $(expr $(cat /scratch1/"${iname}" | wc -l)  - ${len}) -eq 0 ]; do
        len=`cat /scratch1/"${iname}_${i}" | wc -l`
        sleep 3
    done
    sleep 1
    kill $pid

done



sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
