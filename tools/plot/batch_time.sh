#!/bin/bash

bsize=256
apps=(cublas)
size=16384

for app in ${apps[@]}; do
    python3 batch_time.py ../../benchmarks/$app/log_${size}_bsize_${bsize}/${app}_0.txt
done
