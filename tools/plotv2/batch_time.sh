#!/bin/bash

#bsizes=(256 2048)
bsizes=(256)

#apps=(cublas)
#size=16384

apps=(stream)
size=500002816

#apps=(linear)
#size=4194304

for app in ${apps[@]}; do
    for bsize in ${bsizes[@]}; do
        #python3 batch_time.py ../../benchmarksv2/$app/log_${size}_bsize_${bsize}/${app}_0.txt &
        #python3 batch_time.py ../../benchmarksv2/$app/log_pf_${size}_bsize_${bsize}/${app}_0.txt &
        #python3 batch_dist_plot.py ../../benchmarksv2/$app/log_${size}_bsize_${bsize}/${app}_0.txt &
        #python3 batch_dist_plot.py ../../benchmarksv2/$app/log_pf_${size}_bsize_${bsize}/${app}_0.txt &
        true
    done
done
wait


bsize=256

app=cublas
size=16384

app2=stream
size2=500002816

#python3 batch_time.py ../../benchmarksv2/$app/log_${size}_bsize_${bsize}/${app}_0.txt  ../../benchmarksv2/$app2/log_${size2}_bsize_${bsize}/${app2}_0.txt
python3 batch_time.py ../../benchmarksv2/$app2/log_${size2}_bsize_${bsize}/${app2}_0.txt ../../benchmarksv2/$app/log_${size}_bsize_${bsize}/${app}_0.txt
