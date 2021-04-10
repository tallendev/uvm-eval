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

#define MISC_THREAD_BLOCK_SIZE		256

// CUB library is used for reductions
#include "cub/cub.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__ double atomicMax(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(max(val,
                        __longlong_as_double(assumed))));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}

// this kernel zeros out the grid
__global__ void zero_vector_kernel(level_type level, int component_id)
{
  int block = blockIdx.x;

    const int box = level.my_blocks[block].read.box;
          int ilo = level.my_blocks[block].read.i;
          int jlo = level.my_blocks[block].read.j;
          int klo = level.my_blocks[block].read.k;
          int ihi = level.my_blocks[block].dim.i + ilo;
          int jhi = level.my_blocks[block].dim.j + jlo;
          int khi = level.my_blocks[block].dim.k + klo;
    const int jStride = level.my_boxes[box].jStride;
    const int kStride = level.my_boxes[box].kStride;
    const int  ghosts = level.my_boxes[box].ghosts;
    const int     dim = level.my_boxes[box].dim;

    // expand the size of the block to include the ghost zones...
    if(ilo<=  0)ilo-=ghosts;
    if(jlo<=  0)jlo-=ghosts;
    if(klo<=  0)klo-=ghosts;
    if(ihi>=dim)ihi+=ghosts;
    if(jhi>=dim)jhi+=ghosts;
    if(khi>=dim)khi+=ghosts;

    double * __restrict__ grid = level.my_boxes[box].vectors[component_id] + ghosts*(1+jStride+kStride);

    // note here that (ihi - ilo) can be greater than dim becasue of ghost blocks
    int dim_i = (ihi - ilo);
    int i = ilo + threadIdx.x % dim_i;
    if (i >= ihi) return;

    int j_block_stride = MISC_THREAD_BLOCK_SIZE / dim_i;

    for (int j = jlo + threadIdx.x / dim_i; j < jhi; j += j_block_stride)
      for (int k = klo; k < khi; k++) {
        const int ijk = i + j*jStride + k*kStride;
        grid[ijk] = 0.0;
      }
}

// this kernel provides a generic axpy/mul implementation:
// if mul_vectors = 1: c = scale_a * a * b
// if mul_vectors = 0: c = scale_a * a + scale_b * b + shift_a
template <int mul_vectors>
__global__ void axpy_vector_kernel(level_type level, int id_c, double scale_a, double shift_a, double scale_b, int id_a, int id_b)
{
  int block = blockIdx.x;

    const int box = level.my_blocks[block].read.box;
          int ilo = level.my_blocks[block].read.i;
          int jlo = level.my_blocks[block].read.j;
          int klo = level.my_blocks[block].read.k;
          int ihi = level.my_blocks[block].dim.i + ilo;
          int jhi = level.my_blocks[block].dim.j + jlo;
          int khi = level.my_blocks[block].dim.k + klo;
    const int jStride = level.my_boxes[box].jStride;
    const int kStride = level.my_boxes[box].kStride;
    const int  ghosts = level.my_boxes[box].ghosts;
    double * __restrict__ grid_c = level.my_boxes[box].vectors[id_c] + ghosts*(1+jStride+kStride);
    double * __restrict__ grid_a = level.my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);
    double * __restrict__ grid_b = level.my_boxes[box].vectors[id_b] + ghosts*(1+jStride+kStride);

    int dim_i = (ihi - ilo);
    int i = ilo + threadIdx.x % dim_i;
    if (i >= ihi) return;

    int j_block_stride = MISC_THREAD_BLOCK_SIZE / dim_i;

    for (int j = jlo + threadIdx.x / dim_i; j < jhi; j += j_block_stride)
      for (int k = klo; k < khi; k++) {
        const int ijk = i + j*jStride + k*kStride;
        if (mul_vectors)
          grid_c[ijk] = scale_a*grid_a[ijk]*grid_b[ijk];
        else
          grid_c[ijk] = scale_a*grid_a[ijk] + scale_b*grid_b[ijk] + shift_a;
      }
}

// simple coloring kernel, see misc.c for details
__global__ void color_vector_kernel(level_type level, int id_a, int colors_in_each_dim, int icolor, int jcolor, int kcolor)
{
  int block = blockIdx.x;

    const int box = level.my_blocks[block].read.box;
          int ilo = level.my_blocks[block].read.i;
          int jlo = level.my_blocks[block].read.j;
          int klo = level.my_blocks[block].read.k;
          int ihi = level.my_blocks[block].dim.i + ilo;
          int jhi = level.my_blocks[block].dim.j + jlo;
          int khi = level.my_blocks[block].dim.k + klo;
    const int boxlowi = level.my_boxes[box].low.i;
    const int boxlowj = level.my_boxes[box].low.j;
    const int boxlowk = level.my_boxes[box].low.k;
    const int jStride = level.my_boxes[box].jStride;
    const int kStride = level.my_boxes[box].kStride;
    const int  ghosts = level.my_boxes[box].ghosts;
    double * __restrict__ grid = level.my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);

    int dim_i = (ihi - ilo);
    int i = ilo + threadIdx.x % dim_i;
    if (i >= ihi) return;

    int j_block_stride = MISC_THREAD_BLOCK_SIZE / dim_i;

    for (int j = jlo + threadIdx.x / dim_i; j < jhi; j += j_block_stride)
      for (int k = klo; k < khi; k++) {
        double sk=0.0;if( ((k+boxlowk+kcolor)%colors_in_each_dim) == 0 )sk=1.0; // if colors_in_each_dim==1 (don't color), all cells are set to 1.0
        double sj=0.0;if( ((j+boxlowj+jcolor)%colors_in_each_dim) == 0 )sj=1.0;
        double si=0.0;if( ((i+boxlowi+icolor)%colors_in_each_dim) == 0 )si=1.0;
        const int ijk = i + j*jStride + k*kStride;
        grid[ijk] = si*sj*sk;
      }
}

// 0: summation, 1: maximum absolute
template <int red_type>
__global__ void reduction_kernel(level_type level, int id, double *res)
{
  int block = blockIdx.x;

    const int box = level.my_blocks[block].read.box;
          int ilo = level.my_blocks[block].read.i;
          int jlo = level.my_blocks[block].read.j;
          int klo = level.my_blocks[block].read.k;
          int ihi = level.my_blocks[block].dim.i + ilo;
          int jhi = level.my_blocks[block].dim.j + jlo;
          int khi = level.my_blocks[block].dim.k + klo;
    const int jStride = level.my_boxes[box].jStride;
    const int kStride = level.my_boxes[box].kStride;
    const int  ghosts = level.my_boxes[box].ghosts;
    double * __restrict__ grid = level.my_boxes[box].vectors[id] + ghosts*(1+jStride+kStride);

    // accumulate per thread first (multiple elements)
    double thread_val = 0.0;

    int dim_i = (ihi - ilo);
    int i = ilo + threadIdx.x % dim_i;
    if (i < ihi) {
      int j_block_stride = MISC_THREAD_BLOCK_SIZE / dim_i;
      for (int j = jlo + threadIdx.x / dim_i; j < jhi; j += j_block_stride)
        for (int k = klo; k < khi; k++) {
          const int ijk = i + j*jStride + k*kStride;
          double val = grid[ijk];
          switch (red_type) {
          case 0: thread_val += val; break;
          case 1: thread_val = max(thread_val, fabs(val)); break;
          }
        }
     }

  typedef cub::BlockReduce<double, MISC_THREAD_BLOCK_SIZE> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  double block_val;
  switch (red_type) {
  case 0:
    block_val = BlockReduceT(temp_storage).Sum(thread_val);
    if (threadIdx.x == 0) atomicAdd(res, block_val);
    break;
  case 1:
    block_val = BlockReduceT(temp_storage).Reduce(thread_val, cub::Max());
    if (threadIdx.x == 0) atomicMax(res, block_val);
    break;
  }
}

extern "C"
void cuda_zero_vector(level_type d_level, int id)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  zero_vector_kernel<<<grid, block>>>(d_level, id);
  CUDA_ERROR
}

extern "C"
void cuda_scale_vector(level_type d_level, int id_c, double scale_a, int id_a)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  axpy_vector_kernel<0><<<grid, block>>>(d_level, id_c, scale_a, 0.0, 0.0, id_a, id_a);
  CUDA_ERROR
}

extern "C"
void cuda_shift_vector(level_type d_level, int id_c, double shift_a, int id_a)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  axpy_vector_kernel<0><<<grid, block>>>(d_level, id_c, 1.0, shift_a, 0.0, id_a, id_a);
  CUDA_ERROR
}

extern "C"
void cuda_mul_vectors(level_type d_level, int id_c, double scale, int id_a, int id_b)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  axpy_vector_kernel<1><<<grid, block>>>(d_level, id_c, scale, 0.0, 0.0, id_a, id_b);
  CUDA_ERROR
}

extern "C"
void cuda_add_vectors(level_type d_level, int id_c, double scale_a, int id_a, double scale_b, int id_b)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  axpy_vector_kernel<0><<<grid, block>>>(d_level, id_c, scale_a, 0.0, scale_b, id_a, id_b);
  CUDA_ERROR
}

extern "C"
double cuda_sum(level_type d_level, int id)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return 0.0;

  double *d_res;
  double h_res[1];
  CUDA_API_ERROR( cudaMallocManaged((void**)&d_res, sizeof(double), cudaMemAttachGlobal) )
  CUDA_API_ERROR( cudaMemsetAsync(d_res, 0, sizeof(double)) )

  reduction_kernel<0><<<grid, block>>>(d_level, id, d_res);
  CUDA_ERROR

  // sync here to guarantee that the result is updated on GPU
  CUDA_API_ERROR( cudaDeviceSynchronize() )
  h_res[0] = d_res[0];

  CUDA_API_ERROR( cudaFree(d_res) )
  return h_res[0];
}

extern "C"
double cuda_max_abs(level_type d_level, int id)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return 0.0;

  double *d_res;
  double h_res[1];
  CUDA_API_ERROR( cudaMallocManaged((void**)&d_res, sizeof(double), cudaMemAttachGlobal) )
  CUDA_API_ERROR( cudaMemsetAsync(d_res, 0, sizeof(double)) )

  reduction_kernel<1><<<grid, block>>>(d_level, id, d_res);
  CUDA_ERROR

  // sync here to guarantee that the result is updated on GPU
  CUDA_API_ERROR( cudaDeviceSynchronize() )
  h_res[0] = d_res[0];

  CUDA_API_ERROR( cudaFree(d_res) )
  return h_res[0];
}

extern "C"
void cuda_color_vector(level_type d_level, int id_a, int colors_in_each_dim, int icolor, int jcolor, int kcolor)
{
  int block = MISC_THREAD_BLOCK_SIZE;
  int grid = d_level.num_my_blocks;
  if (grid <= 0) return;

  color_vector_kernel<<<grid, block>>>(d_level, id_a, colors_in_each_dim, icolor, jcolor, kcolor);
  CUDA_ERROR
}

