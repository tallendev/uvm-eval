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
template<int LOG_DIM_I, int BLOCK_I, int BLOCK_J, int BLOCK_K, int REBUILD>
__global__ void residual_kernel(level_type level, int res_id, int x_id, int rhs_id, double a, double b){
  const int idim = level.my_blocks[blockIdx.x].dim.i;
  const int jdim = level.my_blocks[blockIdx.x].dim.j;
  const int kdim = min(level.my_blocks[blockIdx.x].dim.k, BLOCK_K);

  // thread boundary conditions
  if(threadIdx.x >= idim || threadIdx.y >= jdim) return;


  ///////////////////// PROLOGUE /////////////////////
  const int box = level.my_blocks[blockIdx.x].read.box;
  const int ilo = level.my_blocks[blockIdx.x].read.i;
  const int jlo = level.my_blocks[blockIdx.x].read.j;
  const int klo = level.my_blocks[blockIdx.x].read.k;
  const int ghosts  = level.my_boxes[box].ghosts;
  const int jStride = level.my_boxes[box].jStride;
  const int kStride = level.my_boxes[box].kStride;
  const double h2inv = 1.0/(level.h*level.h);

        double * __restrict__ rhs      = level.my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  #ifdef USE_HELMHOLTZ
  const double * __restrict__ alpha    = level.my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  #endif
  const double * __restrict__ beta_i   = level.my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  const double * __restrict__ beta_j   = level.my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  const double * __restrict__ beta_k   = level.my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);

        double * __restrict__ res      = level.my_boxes[box].vectors[       res_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  const double * __restrict__ x        = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  ////////////////////////////////////////////////////


  for(int k=0; k<kdim; k++){
    const int ijk = threadIdx.x + threadIdx.y*jStride + k*kStride;

    // apply operator
    const double Ax = apply_op_ijk();

    if (!REBUILD) {
      // residual
      res[ijk] = rhs[ijk] - Ax;
    }
    else {
      // subroutine in rebuild to calculate inverse D
      // repurpose variables: Aii == rhs, sumAbsAij == res
      rhs[ijk] +=      (    x[ijk])*Ax; // add the effect of setting one grid point (i) to 1.0 to Aii
      res[ijk] += fabs((1.0-x[ijk])*Ax);
    }
  }
}
//------------------------------------------------------------------------------------------------------------------------------
#undef  STENCIL_KERNEL
#define STENCIL_KERNEL(log_dim_i, block_i, block_j, block_k) \
  residual_kernel<log_dim_i, block_i, block_j, block_k, 0><<<num_blocks, dim3(block_i, block_j)>>>(level, res_id, x_id, rhs_id, a, b);

extern "C"
void cuda_residual(level_type level, int res_id, int x_id, int rhs_id, double a, double b)
{
  int num_blocks = level.num_my_blocks; if(num_blocks<=0) return;
  int log_dim_i = (int)log2((double)level.dim.i);
  int block_dim_i = min(level.box_dim, BLOCKCOPY_TILE_I);
  int block_dim_k = min(level.box_dim, BLOCKCOPY_TILE_K);

  STENCIL_KERNEL_LEVEL(log_dim_i)
  CUDA_ERROR
}

#undef  STENCIL_KERNEL
#define STENCIL_KERNEL(log_dim_i, block_i, block_j, block_k) \
  residual_kernel<log_dim_i, block_i, block_j, block_k, 1><<<num_blocks, dim3(block_i, block_j)>>>(level, sumAbsAij_id, x_id, Aii_id, a, b);

extern "C"
void cuda_rebuild(level_type level, int x_id, int Aii_id, int sumAbsAij_id, double a, double b)
{
  int num_blocks = level.num_my_blocks; if(num_blocks<=0) return;
  int log_dim_i = (int)log2((double)level.dim.i);
  int block_dim_i = min(level.box_dim, BLOCKCOPY_TILE_I);
  int block_dim_k = min(level.box_dim, BLOCKCOPY_TILE_K);

  STENCIL_KERNEL_LEVEL(log_dim_i)
  CUDA_ERROR
}

