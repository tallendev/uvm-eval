#!/bin/bash -xe

sudo stap --vp 01 ./profile.stp ./matrixMul2
#sudo stap --vp 01 ./profile.stp "./matrixMul2 -Ha=16384 -Hb=16384 -Wa=16384 -Wb=16384"
