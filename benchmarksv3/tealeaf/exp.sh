#!/bin/bash -x

module load openmpi_gcc
module load cuda

cores=40
procs=4
cpe=$(($cores/$procs))

export OMP_NUM_THREADS=$cpe
export OMP_MAX_THREADS=100 #$(($cores/$procs))
#export KMP_AFFINITY=compact
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1



rm -f runtime.dat
#valgrind --track-origins=yes ./tealeaf
#nvprof ./tealeaf
#mpirun  -np 1 ./snap snap.in snap.out

for ((i=1; $i < 128; i=$i*2)); do
    val=$((448*$i))
    sed -i "s/\(x_cells.*\)= [0-9]\+/\1= $val/" tea.in
    sed -i "s/\(y_cells.*\)= [0-9]\+/\1= $val/" tea.in
    echo $val

    mpirun  -x OMP_NUM_THREADS -x OMP_MAX_THREADS -x OMP_PROC_BIND -x OMP_PLACES\
        --use-hwthread-cpus --map-by node:PE=$cpe  --report-bindings \
        -np $procs -N 1 \
        /usr/local/cuda-9.0/bin/nvprof --csv -f --log-file tealeaf.%q{OMPI_COMM_WORLD_RANK}-$val.log \
        ./tealeaf &> runout.log
    grep "Wallclock:" runout.log | awk -F' ' -v val=$val '{ print val","$2}' >> runtime.dat 
    echo $?
done
#./snap snap.in snap.out
#-o tealeaf.%q{OMPI_COMM_WORLD_RANK}.nvprof \
