# Navigation:
## benchmarksv3:
    - Contains all benchmarks in paper and some additional examples
    - "bsize-scripts" contains scripts for generating batch-oriented data.
    - "fault-scripts" contains scripts for generating fault-based data.
## drivers:
    - `batchd` includes batch-based modifications
    - `faults` includes fault-based data
## tools:
    - `syslogger` contains the tool for parsing data out of the system log
    - `plotv2` contains .sh scripts that operate the .py scripts for reproducing plots and analysis from the paper if data is available



## Applications
The scripts will reproduce the experiments as they were performed for use in the paper, including data not used or omitted for space.
We will document them in detail here:

### sgemm
benchmarks/cublas
description: sgemm from CUBLAS libary.
build: make
run: ./matrixMul2 -wA=${psize} -hA=${psize} -wB=${psize} -hB=${psize}
data sets: programmatically generated - 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768

### stream
benchmarks/stream
description: stream triad from BabelStream
build: make
run: ./cuda-stream -n 1 -s $psize --triad-only
data sets: programmatically generated - 125000704, 250001408, 375002112, 500002816, 625003520, 750004224

### regular access pattern
benchmarks/linear
description: accesses pages as sequentially as possible, with full data coalescing
build: make DEFS=-DPNUM=$psize
run: ./page
data sets: programmatically generated - 1048576, 8388608


### random access pattern
benchmarks/random
description: accesses each page randomly, uniquely, with the same data set as linear/regular
build: make DEFS=-DPNUM=$psize
run: ./page
data sets: programmatically generated - 1048576, 8388608

### hpgmg
benchmarks/hpgmg
description: NVIDIA UVM hpgmg - https://bitbucket.org/nsakharnykh/hpgmg-cuda/src/master/
build: ./build.sh (complex options for ./configure, make)
run: ./build/bin/hpgmg-fv [log2_box_dim]  [target_boxes_per_rank]
data sets: programmatically generated - "7 7", "8 8"

### cufft
benchmarks/cufft
description: cufft example - https://github.com/NVIDIA/cuda-samples/tree/master/Samples/simpleCUFFT
build: make
run: ./simpleCUFFT
data sets: programmatically generated - 80000000 

### tealeaf
benchmarks/tealeaf
description: tealeaf - https://github.com/UoB-HPC/TeaLeaf/tree/master/2d
build: make
run: ./tealeaf
data sets: "tea.in" is default input file - "smaller-tea.in", "smaller-tea.in", "large-tea.in" used for experiments by overwriting tea.in
