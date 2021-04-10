#!/bin/bash -xe 

sudo ./try.stp > trace.log &

./matrixMul2 -Ha=16384 -Hb=16384 -Wa=16384 -Wb=16384

sudo killall /usr/bin/stap
