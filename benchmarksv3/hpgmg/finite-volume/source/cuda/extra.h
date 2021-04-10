/*
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//------------------------------------------------------------------------------------------------------------------------------
// NVIDIA Profiler
#ifdef USE_PROFILE
#include <cuda_profiler_api.h>
#define CUDA_PROFILER_START_ON_LEVEL(start) { if(start){cudaProfilerStart();} }
#define CUDA_PROFILER_STOP_ON_LEVEL(stop)   { if(stop){cudaProfilerStop();cudaDeviceReset();exit(0);} }
#else
#define CUDA_PROFILER_START_ON_LEVEL(start) { }
#define CUDA_PROFILER_STOP_ON_LEVEL(stop)   { }
#endif

//------------------------------------------------------------------------------------------------------------------------------
// NVTX
#ifdef USE_NVTX
#include <cuda_profiler_api.h>
#include "nvToolsExt.h"
#define NVTX_PUSH(name,cid) { \
        cudaDeviceSynchronize(); \
	uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff }; \
	int num_colors = sizeof(colors)/sizeof(uint32_t); \
	int color_id = cid; \
	color_id = color_id%num_colors;\
	nvtxEventAttributes_t eventAttrib = {0}; \
	eventAttrib.version = NVTX_VERSION; \
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
	eventAttrib.colorType = NVTX_COLOR_ARGB; \
	eventAttrib.color = colors[color_id]; \
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
	eventAttrib.message.ascii = name; \
	nvtxRangePushEx(&eventAttrib); \
}
#define NVTX_POP { cudaDeviceSynchronize(); nvtxRangePop(); }
#else
#define NVTX_PUSH(name,cid) {}
#define NVTX_POP {}
#endif

//------------------------------------------------------------------------------------------------------------------------------
// CUDA errors
#ifdef USE_ERROR
#include <execinfo.h>
// wrapper for CUDA API errors
#define CUDA_API_ERROR(func) { \
  cudaError_t status = func; \
  if (status != cudaSuccess) { \
    printf("CUDA ERROR in %s, line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    void *callstack[128]; \
    int frames = backtrace(callstack, 128); \
    char **strs = backtrace_symbols(callstack, frames); \
    for (int i = 0; i < frames; i++) \
      printf("[bt] %s\n", strs[i]); \
    free(strs); \
    cudaDeviceReset(); \
    exit(-1); \
  } \
}
// wrapper for CUDA kernel errors
#define CUDA_ERROR 	     { \
  CUDA_API_ERROR( cudaDeviceSynchronize() ); \
}
#else
#define CUDA_API_ERROR(func) { func; }
#define CUDA_ERROR           { }
#endif

//------------------------------------------------------------------------------------------------------------------------------
// Template stencil kernels at different levels

// select appropriate block size
#define STENCIL_KERNEL_TILE(log_dim_i) 	\
  if(block_dim_i == BLOCKCOPY_TILE_I)	STENCIL_KERNEL(log_dim_i,      BLOCKCOPY_TILE_I, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
  else if(block_dim_i <= 1)		STENCIL_KERNEL(log_dim_i,   1, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
  else if(block_dim_i <= 2)		STENCIL_KERNEL(log_dim_i,   2, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
  else if(block_dim_i <= 4)		STENCIL_KERNEL(log_dim_i,   4, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
  else if(block_dim_i <= 8)		STENCIL_KERNEL(log_dim_i,   8, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
  else if(block_dim_i <= 16)		STENCIL_KERNEL(log_dim_i,  16, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
  else if(block_dim_i <= 32)		STENCIL_KERNEL(log_dim_i,  32, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
  else if(block_dim_i <= 64)		STENCIL_KERNEL(log_dim_i,  64, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
  else if(block_dim_i <= 128)		STENCIL_KERNEL(log_dim_i, 128, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
  else if(block_dim_i <= 256)		STENCIL_KERNEL(log_dim_i, 256, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
  else { printf("CUDA ERROR: tile dimension %i is not supported in the GPU path, please update the macros!\n", log_dim_i); exit(1); }

// maximum supported level can have 2^10 dimension
#define STENCIL_KERNEL_LEVEL(log_dim_i)	      \
  switch(log_dim_i){ 		      	      \
  case 0:  { STENCIL_KERNEL_TILE(0)  break; } \
  case 1:  { STENCIL_KERNEL_TILE(1)  break; } \
  case 2:  { STENCIL_KERNEL_TILE(2)  break; } \
  case 3:  { STENCIL_KERNEL_TILE(3)  break; } \
  case 4:  { STENCIL_KERNEL_TILE(4)  break; } \
  case 5:  { STENCIL_KERNEL_TILE(5)  break; } \
  case 6:  { STENCIL_KERNEL_TILE(6)  break; } \
  case 7:  { STENCIL_KERNEL_TILE(7)  break; } \
  case 8:  { STENCIL_KERNEL_TILE(8)  break; } \
  case 9:  { STENCIL_KERNEL_TILE(9)  break; } \
  case 10: { STENCIL_KERNEL_TILE(10) break; } \
  case 11: { STENCIL_KERNEL_TILE(11) break; } \
  case 12: { STENCIL_KERNEL_TILE(12) break; } \
  case 13: { STENCIL_KERNEL_TILE(13) break; } \
  case 14: { STENCIL_KERNEL_TILE(14) break; } \
  case 15: { STENCIL_KERNEL_TILE(15) break; } \
  case 16: { STENCIL_KERNEL_TILE(16) break; } \
  case 17: { STENCIL_KERNEL_TILE(17) break; } \
  case 18: { STENCIL_KERNEL_TILE(18) break; } \
  case 19: { STENCIL_KERNEL_TILE(19) break; } \
  case 20: { STENCIL_KERNEL_TILE(20) break; } \
  default: { printf("CUDA ERROR: level size 2^%i is not supported in the GPU path, please update the macros!\n", log_dim_i); exit(1); }}

//------------------------------------------------------------------------------------------------------------------------------
// Template kernels at different levels

// maximum supported level can have 2^10 dimension
#define KERNEL_LEVEL(log_dim, block_type)    \
  switch(log_dim){                           \
  case 0:  { KERNEL(0, block_type)  break; } \
  case 1:  { KERNEL(1, block_type)  break; } \
  case 2:  { KERNEL(2, block_type)  break; } \
  case 3:  { KERNEL(3, block_type)  break; } \
  case 4:  { KERNEL(4, block_type)  break; } \
  case 5:  { KERNEL(5, block_type)  break; } \
  case 6:  { KERNEL(6, block_type)  break; } \
  case 7:  { KERNEL(7, block_type)  break; } \
  case 8:  { KERNEL(8, block_type)  break; } \
  case 9:  { KERNEL(9, block_type)  break; } \
  case 10: { KERNEL(10, block_type) break; } \
  case 11: { KERNEL(11, block_type) break; } \
  case 12: { KERNEL(12, block_type) break; } \
  case 13: { KERNEL(13, block_type) break; } \
  case 14: { KERNEL(14, block_type) break; } \
  case 15: { KERNEL(15, block_type) break; } \
  case 16: { KERNEL(16, block_type) break; } \
  case 17: { KERNEL(17, block_type) break; } \
  case 18: { KERNEL(18, block_type) break; } \
  case 19: { KERNEL(19, block_type) break; } \
  case 20: { KERNEL(20, block_type) break; } \
  default: { printf("CUDA ERROR: level size 2^%i is not supported in the GPU path, please update the macros!\n", log_dim); exit(1); }}
