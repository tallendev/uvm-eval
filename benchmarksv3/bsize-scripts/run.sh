#!/bin/bash -xe

sbatch hpgmg.sh
sbatch cublas.sh
sbatch cublas-long.sh
sbatch stream.sh
sbatch linear.sh
sbatch random.sh
sbatch cufft.sh
sbatch hpgmg-p.sh
sbatch tealeaf.sh
sbatch tealeaf-parallel.sh
