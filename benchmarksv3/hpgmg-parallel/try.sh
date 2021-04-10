#!/bin/bash -xe 

sudo ./try.stp > trace.log &


./build/bin/hpgmg-fv 7 7

sudo killall /usr/bin/stap
