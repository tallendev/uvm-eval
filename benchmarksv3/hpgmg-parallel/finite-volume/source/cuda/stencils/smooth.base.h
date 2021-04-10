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

    printf("SCALAB smooth base\n");

  ///////////////////// PROLOGUE /////////////////////
  const int box = level.my_blocks[blockIdx.x].read.box;
  const int ilo = level.my_blocks[blockIdx.x].read.i;
  const int jlo = level.my_blocks[blockIdx.x].read.j;
  const int klo = level.my_blocks[blockIdx.x].read.k;
  const int ghosts  = level.my_boxes[box].ghosts;
  const int jStride = level.my_boxes[box].jStride;
  const int kStride = level.my_boxes[box].kStride;
  const double h2inv = 1.0/(level.h*level.h);

  const double * __restrict__ rhs      = level.my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  #ifdef USE_HELMHOLTZ
  const double * __restrict__ alpha    = level.my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  #endif
  const double * __restrict__ beta_i   = level.my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  const double * __restrict__ beta_j   = level.my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  const double * __restrict__ beta_k   = level.my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  #ifdef USE_L1JACOBI
  const double * __restrict__ Dinv     = level.my_boxes[box].vectors[VECTOR_L1INV ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  #else
  const double * __restrict__ Dinv     = level.my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  #endif

        double * __restrict__ xo;
  const double * __restrict__ xp;
  const double * __restrict__ x;
                   if((s&1)==0){x      = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
                                xp     = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
                                xo     = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);}
                           else{x      = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
                                xp     = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
                                xo     = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);}

  #ifdef USE_CHEBY
  const double c1 = c[s%CHEBYSHEV_DEGREE];
  const double c2 = d[s%CHEBYSHEV_DEGREE];
  #elif USE_GSRB
  const int color000 = (level.my_boxes[box].low.i^level.my_boxes[box].low.j^level.my_boxes[box].low.k^s)&1;
  #endif
  ////////////////////////////////////////////////////


  for(int k=0; k<kdim; k++){
    const int ijk = threadIdx.x + threadIdx.y*jStride + k*kStride;


    // apply operator
    const double Ax = apply_op_ijk();


    ///////////////////// SMOOTHER /////////////////////
    #ifdef USE_CHEBY
    const double lambda = Dinv_ijk();
    xo[ijk] = X(ijk) + c1*(X(ijk)-xp[ijk]) + c2*lambda*(rhs[ijk]-Ax);


    #elif USE_JACOBI
    const double lambda = Dinv_ijk();
    xo[ijk] = X(ijk) + (0.6666666666666666667)*lambda*(rhs[ijk]-Ax);


    #elif USE_L1JACOBI
    const double lambda = Dinv_ijk();
    xo[ijk] = X(ijk) + lambda*(rhs[ijk]-Ax);


    #elif USE_SYMGS
    // add code here


    #elif USE_GSRB
    const double * __restrict__ RedBlack = level.RedBlack_FP + ghosts*(1+jStride) + (((k+klo)^color000)&1)*kStride + (ilo + jlo*jStride);
    const double lambda = Dinv_ijk();
    const int ij  = threadIdx.x + threadIdx.y*jStride;
    xo[ijk] = X(ijk) + RedBlack[ij]*lambda*(rhs[ijk]-Ax);
    #endif
    ////////////////////////////////////////////////////
  }
}
//------------------------------------------------------------------------------------------------------------------------------
#define STENCIL_KERNEL(log_dim_i, block_i, block_j, block_k) \
  smooth_kernel<log_dim_i, block_i, block_j, block_k><<<num_blocks, dim3(block_i, block_j)>>>(level, x_id, rhs_id, a, b, s, c, d);

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
