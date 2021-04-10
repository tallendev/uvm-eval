#!/bin/bash -xe

module load julia
julia precompiler.jl
mv sys_plots.so sys_plots_`hostname`.so
