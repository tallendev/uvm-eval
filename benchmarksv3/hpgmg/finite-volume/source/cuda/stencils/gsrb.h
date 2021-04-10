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
template<int LOG_DIM_I, int BLOCK_I, int BLOCK_J, int BLOCK_K>
__global__ void smooth_kernel(level_type level, int x_id, int rhs_id, double a, double b, int s, double *c, double *d){
  const int idim = level.my_blocks[blockIdx.x].dim.i;
  const int jdim = level.my_blocks[blockIdx.x].dim.j;
  const int kdim = min(level.my_blocks[blockIdx.x].dim.k, BLOCK_K);

  // thread boundary conditions
  if(threadIdx.x >= idim || threadIdx.y >= jdim) return;

  printf("SCALAB gsrb\n");

  ///////////////////// PROLOGUE /////////////////////
  const int box = level.my_blocks[blockIdx.x].read.box;
  const int ilo = level.my_blocks[blockIdx.x].read.i;
  const int jlo = level.my_blocks[blockIdx.x].read.j;
  const int klo = level.my_blocks[blockIdx.x].read.k;
  const int ghosts  = level.my_boxes[box].ghosts;
  const int jStride = level.my_boxes[box].jStride;
  const int kStride = level.my_boxes[box].kStride;
  const double h2inv = 1.0/(level.h*level.h);

  const double * __restrict__ rhs      = level.my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride);
  #ifdef USE_HELMHOLTZ
  const double * __restrict__ alpha    = level.my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);
  #endif
  const double * __restrict__ beta_i   = level.my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride);
  const double * __restrict__ beta_j   = level.my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride);
  const double * __restrict__ beta_k   = level.my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride);
  const double * __restrict__ Dinv     = level.my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride);

  #ifdef GSRB_OOP
        double * __restrict__ xo;
  const double * __restrict__ x;
                   if((s&1)==0){x      = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);
                                xo     = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);}
                           else{x      = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);
                                xo     = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);}
  #else
        double * __restrict__ xo       = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);
  const double * __restrict__ x        = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);
  #endif
  const int color000 = (level.my_boxes[box].low.i^level.my_boxes[box].low.j^level.my_boxes[box].low.k^s)&1;
  ////////////////////////////////////////////////////


  int i = ilo + threadIdx.x;
  int j = jlo + threadIdx.y;


  #ifdef GSRB_FP
  for(int k=klo; k<(klo+kdim); k++){
    const int ijk = i + j*jStride + k*kStride;
    const double * __restrict__ RedBlack = level.RedBlack_FP + ghosts*(1+jStride) + ((k^color000)&1)*kStride;
    const double Ax = apply_op_ijk();
    const double lambda = Dinv_ijk();
    const int ij = i + j*jStride;
    xo[ijk] = X(ijk) + RedBlack[ij]*lambda*(rhs[ijk]-Ax);
    //x_np1[ijk] = ((i^j^k^color000)&1) ? x_n[ijk] : x_n[ijk] + lambda*(rhs[ijk]-Ax);
  }
  

  #elif GSRB_STRIDE2
  for(int k=klo; k<klo+kdim; k++){
    #ifdef GSRB_OOP
    // out-of-place must copy old value...
    i = ilo +!((ilo^j^k^color000)&1) + threadIdx.x*2;if(i < ilo+idim){ // stride-2 GSRB
    const int ijk = i + j*jStride + k*kStride;
    xo[ijk] = X(ijk);
    }
    #endif
    i = ilo + ((ilo^j^k^color000)&1) + threadIdx.x*2;if(i < ilo+idim){ // stride-2 GSRB
    const int ijk = i + j*jStride + k*kStride;
    const double Ax = apply_op_ijk();
    const double lambda = Dinv_ijk();
    xo[ijk] = X(ijk) + lambda*(rhs[ijk]-Ax);
    }
  }

  #elif GSRB_BRANCH
  for(int k=klo; k<klo+kdim; k++){
    const int ijk = i + j*jStride + k*kStride;
    if(((i^j^k^color000^1)&1)){ // looks very clean when [0] is i,j,k=0,0,0
    const double Ax = apply_op_ijk();
    const double lambda = Dinv_ijk();
    xo[ijk] = X(ijk) + lambda*(rhs[ijk]-Ax);
    #ifdef GSRB_OOP
    }else{ xo[ijk] = X(ijk); // copy old value when sweep color != cell color
    #endif
    }
  }


  #else
  #error no GSRB implementation was specified
  #endif
}
//------------------------------------------------------------------------------------------------------------------------------
#ifdef GSRB_STRIDE2
  #define STENCIL_KERNEL(log_dim_i, block_i, block_j, block_k) \
  smooth_kernel<log_dim_i, block_i/2, block_j, block_k><<<num_blocks, dim3(block_i/2, block_j)>>>(level, x_id, rhs_id, a, b, s, c, d);
#else
  #define STENCIL_KERNEL(log_dim_i, block_i, block_j, block_k) \
  smooth_kernel<log_dim_i, block_i, block_j, block_k><<<num_blocks, dim3(block_i, block_j)>>>(level, x_id, rhs_id, a, b, s, c, d);
#endif

extern "C"
void cuda_smooth(level_type level, int x_id, int rhs_id, double a, double b, int s, double *c, double *d)
{
  int num_blocks = level.num_my_blocks; if(num_blocks<=0) return;
  int log_dim_i = (int)log2((double)level.dim.i);
  int block_dim_i = min(level.box_dim, BLOCKCOPY_TILE_I);
  int block_dim_k = min(level.box_dim, BLOCKCOPY_TILE_K);

  CUDA_PROFILER_START_ON_LEVEL(log_dim_i==8)

  STENCIL_KERNEL_LEVEL(log_dim_i)
  CUDA_ERROR

  CUDA_PROFILER_STOP_ON_LEVEL(log_dim_i==8)
}
