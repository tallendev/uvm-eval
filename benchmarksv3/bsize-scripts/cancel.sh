#!/bin/bash -xe

for i in `squeue | awk '{ if (NR > 1) print $1}'`; do 
    scancel $i; 
done
