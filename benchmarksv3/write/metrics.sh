#!/bin/bash

#nvprof \
#    --metrics stall_constant_memory_dependency,stall_exec_dependency,stall_inst_fetch,stall_memory_dependency,stall_memory_throttle,stall_not_selected,stall_other,stall_pipe_busy,stall_sleeping,stall_sync,stall_texture \
#    ./abc

nvprof --metrics stall_memory_dependency ./abc
