#!/bin/bash -xe

sbatch cublas.sh
sbatch stream.sh
sbatch linear.sh
sbatch random.sh
sbatch gs.sh
sbatch hpgmg.sh
sbatch hpgmg-p.sh
