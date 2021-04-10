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
__launch_bounds__(128, 4) // force 25% occupancy on Kepler/Maxwell
__global__ void smooth_kernel(level_type level, int x_id, int rhs_id, double a, double b, int s, double *c, double *d){
  const int idim = level.my_blocks[blockIdx.x].dim.i;
  const int jdim = level.my_blocks[blockIdx.x].dim.j;
  const int kdim = min(level.my_blocks[blockIdx.x].dim.k, BLOCK_K);

  // thread boundary conditions
  if(threadIdx.x >= idim || threadIdx.y >= jdim) return;

    //printf("SCALAB smooth  reg fv4.h\n");

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
  #ifdef USE_CHEBY
  const double * __restrict__ xp;
  #endif
  const double * __restrict__ x;
                   if((s&1)==0){x      = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  #ifdef USE_CHEBY
                                xp     = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  #endif
                                xo     = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);}
                           else{x      = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  #ifdef USE_CHEBY
                                xp     = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
  #endif
                                xo     = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);}

  #ifdef USE_CHEBY
  const double c1 = c[s%CHEBYSHEV_DEGREE];
  const double c2 = d[s%CHEBYSHEV_DEGREE];
  #elif USE_GSRB
  const int color000 = (level.my_boxes[box].low.i^level.my_boxes[box].low.j^level.my_boxes[box].low.k^s)&1;
  #endif
  ////////////////////////////////////////////////////


  // store k and k-1 planes into registers
  int ijk = threadIdx.x + threadIdx.y*jStride;
  double xc1,xl1,xr1,xu1,xd1,xc0,xl0,xr0,xu0,xd0,xc2,xl2,xr2,xu2,xd2;
  double xlu,xld,xru,xrd,xll,xrr,xuu,xdd,xbb,xff;
  xc1 = X(ijk);
  xl1 = X(ijk-1);
  xr1 = X(ijk+1);
  xu1 = X(ijk-jStride);
  xd1 = X(ijk+jStride);
  xc0 = X(ijk-kStride);
  xl0 = X(ijk-1-kStride);
  xr0 = X(ijk+1-kStride);
  xu0 = X(ijk-jStride-kStride);
  xd0 = X(ijk+jStride-kStride);
  #ifdef TEX // tex-op
  xll = X(ijk-2);
  xrr = X(ijk+2);
  xuu = X(ijk-2*jStride);
  xdd = X(ijk+2*jStride);
  xbb = X(ijk-2*kStride);
  xff = X(ijk+2*kStride);
  #endif
  double bkc1,bkl1,bkr1,bku1,bkd1,bkc2,bkl2,bkr2,bku2,bkd2;
  bkc1 = BK(ijk);
  bkl1 = BK(ijk-1);
  bkr1 = BK(ijk+1);
  bku1 = BK(ijk-jStride);
  bkd1 = BK(ijk+jStride);
  double bic1,bir1,bic0,bir0,bic2,bir2;
  double biu,bid,bird,biru;
  bic1 = BI(ijk);
  bir1 = BI(ijk+1);
  bic0 = BI(ijk-kStride);
  bir0 = BI(ijk+1-kStride);
  double bjc1,bjd1,bjc0,bjd0,bjc2,bjd2;
  double bjl,bjr,bjld,bjrd;
  bjc1 = BJ(ijk);
  bjd1 = BJ(ijk+jStride);
  bjc0 = BJ(ijk-kStride);
  bjd0 = BJ(ijk+jStride-kStride);

  for(int k=0; k<kdim; k++){
    ijk = threadIdx.x + threadIdx.y*jStride + k*kStride;
    // store k+1 plane and k cells into registers
    xc2 = X(ijk+kStride);
    xl2 = X(ijk-1+kStride);
    xr2 = X(ijk+1+kStride);
    xu2 = X(ijk-jStride+kStride);
    xd2 = X(ijk+jStride+kStride);
    xlu = X(ijk-1-jStride);
    xld = X(ijk-1+jStride);
    xru = X(ijk+1-jStride);
    xrd = X(ijk+1+jStride);
    #ifndef TEX // not tex-op
    xll = X(ijk-2);
    xrr = X(ijk+2);
    xuu = X(ijk-2*jStride);
    xdd = X(ijk+2*jStride);
    xbb = X(ijk-2*kStride);
    xff = X(ijk+2*kStride);
    #endif

    bkc2 = BK(ijk+kStride);
    bkl2 = BK(ijk-1+kStride);
    bkr2 = BK(ijk+1+kStride);
    bku2 = BK(ijk-jStride+kStride);
    bkd2 = BK(ijk+jStride+kStride);

    bic2 = BI(ijk+kStride);
    bir2 = BI(ijk+1+kStride);
    biu  = BI(ijk-jStride);
    bid  = BI(ijk+jStride);
    bird = BI(ijk+1+jStride);
    biru = BI(ijk+1-jStride);

    bjc2 = BJ(ijk+kStride);
    bjd2 = BJ(ijk+jStride+kStride);
    bjl  = BJ(ijk-1);
    bjr  = BJ(ijk+1);
    bjld = BJ(ijk-1+jStride);
    bjrd = BJ(ijk+1+jStride);


    // apply operator
    const double Ax  =
    #ifdef USE_HELMHOLTZ
    a*alpha[ijk]*xc1
    #endif
    -b*h2inv*(
    STENCIL_TWELFTH*(
    + bic1 * ( 15.0*(xl1-xc1) - (xll-xr1) )
    + bir1 * ( 15.0*(xr1-xc1) - (xrr-xl1) )
    + bjc1 * ( 15.0*(xu1-xc1) - (xuu-xd1) )
    + bjd1 * ( 15.0*(xd1-xc1) - (xdd-xu1) )
    + bkc1 * ( 15.0*(xc0-xc1) - (xbb-xc2) )
    + bkc2 * ( 15.0*(xc2-xc1) - (xff-xc0) ) )

    + 0.25*STENCIL_TWELFTH*(
    + (bid  - biu ) * (xld - xd1 - xlu + xu1)
    + (bic2 - bic0) * (xl2 - xc2 - xl0 + xc0)
    + (bjr  - bjl ) * (xru - xr1 - xlu + xl1)
    + (bjc2 - bjc0) * (xu2 - xc2 - xu0 + xc0)
    + (bkr1 - bkl1) * (xr0 - xr1 - xl0 + xl1)
    + (bkd1 - bku1) * (xd0 - xd1 - xu0 + xu1)

    + (bird - biru) * (xrd - xd1 - xru + xu1)
    + (bir2 - bir0) * (xr2 - xc2 - xr0 + xc0)
    + (bjrd - bjld) * (xrd - xr1 - xld + xl1)
    + (bjd2 - bjd0) * (xd2 - xc2 - xd0 + xc0)
    + (bkr2 - bkl2) * (xr2 - xr1 - xl2 + xl1)
    + (bkd2 - bku2) * (xd2 - xd1 - xu2 + xu1) )
    );


    ///////////////////// SMOOTHER /////////////////////
    #ifdef USE_CHEBY
    const double lambda = Dinv_ijk();
    xo[ijk] = xc1 + c1*(xc1-xp[ijk]) + c2*lambda*(rhs[ijk]-Ax);


    #elif USE_JACOBI
    const double lambda = Dinv_ijk();
    xo[ijk] = xc1 + (0.6666666666666666667)*lambda*(rhs[ijk]-Ax);


    #elif USE_L1JACOBI
    const double lambda = Dinv_ijk();
    xo[ijk] = xc1 + lambda*(rhs[ijk]-Ax);


    #elif USE_SYMGS
    // add code here


    #elif USE_GSRB
    const double * __restrict__ RedBlack = level.RedBlack_FP + ghosts*(1+jStride) + (((k+klo)^color000)&1)*kStride + (ilo + jlo*jStride);
    const double lambda = Dinv_ijk();
    const int ij  = threadIdx.x + threadIdx.y*jStride;
    xo[ijk] = xc1 + RedBlack[ij]*lambda*(rhs[ijk]-Ax);
    #endif
    ////////////////////////////////////////////////////


    // store k+1 plane into registers
    #ifdef TEX  // tex-op
    // why does this give such a good speedup?
    if (k<kdim-1) {
    xll = X(ijk-2+kStride);
    xrr = X(ijk+2+kStride);
    xuu = X(ijk-2*jStride+kStride);
    xdd = X(ijk+2*jStride+kStride);
    xbb = X(ijk-2*kStride+kStride);
    xff = X(ijk+2*kStride+kStride);
    }
    #endif
    // update k and k-1 planes in registers
    xc0 = xc1;  xc1 = xc2;
    xl0 = xl1;  xl1 = xl2;
    xr0 = xr1;  xr1 = xr2;
    xu0 = xu1;  xu1 = xu2;
    xd0 = xd1;  xd1 = xd2;

    bkc1 = bkc2;
    bkl1 = bkl2;
    bkr1 = bkr2;
    bku1 = bku2;
    bkd1 = bkd2;

    bic0 = bic1;  bic1 = bic2;
    bir0 = bir1;  bir1 = bir2;

    bjc0 = bjc1;  bjc1 = bjc2;
    bjd0 = bjd1;  bjd1 = bjd2;
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
