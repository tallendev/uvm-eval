#!/bin/bash -xe
# example:
# if ../cublas is full of raw syslogger files:
# ./.sh ../data/cublas/*

for var in "$@"; do
    (out=${var}.txt
    time awk -F';' '{print $2}' ${var} > ${var}.scratch1
    tail -70 ${var}.scratch1 | grep 'uvm range destroy' | awk -F' ' '{print $6, $7}' > $out 
    sort -n $out | sponge $out
    grep '^[sbfpe],' ${var}.scratch1 >> $out
    rm ${var}.scratch1)&
done
wait
