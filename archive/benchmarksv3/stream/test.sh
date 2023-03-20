

psize=500002816

echo "sequential kernel"
module load cuda
cd /home/tnallen/cuda11/uvmmodel-NVIDIA-Linux-x86_64-450.51.05/kernel
make
sudo make modules_install
cd -

sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
sudo rmmod -f nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
./cuda-stream -n 1 -s $psize --triad-only


echo "parallel kernel"
cd /home/tnallen/cuda11/parallel-NVIDIA-Linux-x86_64-450.51.05/kernel
make
sudo make modules_install
cd -

sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
sudo rmmod -f nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

./cuda-stream -n 1 -s $psize --triad-only
