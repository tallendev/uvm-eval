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
//__launch_bounds__((BLOCKCOPY_TILE_I * BLOCKCOPY_TILE_J), RESIDUAL_MIN_BLOCKS_PER_SM)
__global__ void chebyshev_kernel(level_type level, int x_id, int rhs_id, double a, double b, int s, double *chebyshev_c1, double *chebyshev_c2){
      extern __shared__ double l_flux[];
      int block = blockIdx.x;

      double * __restrict__ flux_i = l_flux;
      double * __restrict__ flux_j = l_flux + (BLOCK_I+1)*(BLOCKCOPY_TILE_J+1)*(1);
      double * __restrict__ flux_k = l_flux + (BLOCK_I+1)*(BLOCKCOPY_TILE_J+1)*(2);

      double bh2invSTENCIL_TWELFTH = b*STENCIL_TWELFTH/(level.h*level.h);


      const int box  = level.my_blocks[block].read.box;
      const int ilo  = level.my_blocks[block].read.i;
      const int jlo  = level.my_blocks[block].read.j;
      const int klo  = level.my_blocks[block].read.k;
      const int idim = level.my_blocks[block].dim.i;
      const int jdim = level.my_blocks[block].dim.j;
      const int kdim = level.my_blocks[block].dim.k;

      // thread exit conditions...
      //if(threadIdx.x >= idim || threadIdx.y >= jdim) return;
      int bounds = (threadIdx.x<idim && threadIdx.y<jdim);

      const int ghosts  = level.my_boxes[box].ghosts;
      const int jStride = level.my_boxes[box].jStride;
      const int kStride = level.my_boxes[box].kStride;
      const int flux_jStride = (idim+1);
      const int flux_kStride = (BLOCKCOPY_TILE_J+1)*(idim+1);
      const double * __restrict__ rhs      = level.my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
#ifdef USE_HELMHOLTZ
      const double * __restrict__ alpha    = level.my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
#endif
      const double * __restrict__ beta_i   = level.my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
      const double * __restrict__ beta_j   = level.my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
      const double * __restrict__ beta_k   = level.my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
      const double * __restrict__ Dinv     = level.my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
#ifdef STENCIL_FUSE_BC
      const double * __restrict__ valid    = level.my_boxes[box].vectors[VECTOR_VALID ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
#endif
            double * __restrict__ x_np1;
      const double * __restrict__ x_n;
      const double * __restrict__ x_nm1;
                       if((s&1)==0){x_n    = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
                                    x_nm1  = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
                                    x_np1  = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);}
                               else{x_n    = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
                                    x_nm1  = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);
                                    x_np1  = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride) + (ilo + jlo*jStride + klo*kStride);}
      const double c1 = chebyshev_c1[s%CHEBYSHEV_DEGREE]; // limit polynomial to degree CHEBYSHEV_DEGREE.
      const double c2 = chebyshev_c2[s%CHEBYSHEV_DEGREE]; // limit polynomial to degree CHEBYSHEV_DEGREE.


      int i=threadIdx.x,j=threadIdx.y,k;
      int ijk,flux_ij;
      for(k=0;k<kdim;k++){
        double * __restrict__ flux_klo = flux_k + ((k  )&0x1)*flux_kStride;
        double * __restrict__ flux_khi = flux_k + ((k+1)&0x1)*flux_kStride;
	if(bounds){
        // calculate flux_i and flux_j
              ijk = i + j*jStride + k*kStride;
          flux_ij = i + j*flux_jStride;
          flux_i[flux_ij] = -bh2invSTENCIL_TWELFTH*(
                 (beta_i[ijk        ]) * ( 15.0*(x_n[ijk-1      ]-x_n[ijk]) - (x_n[ijk-2              ]-x_n[ijk+1      ]) ) + 
            0.25*(beta_i[ijk+jStride]-beta_i[ijk-jStride]) * (x_n[ijk-1      +jStride]-x_n[ijk+jStride]-x_n[ijk-1      -jStride]+x_n[ijk-jStride]) + 
            0.25*(beta_i[ijk+kStride]-beta_i[ijk-kStride]) * (x_n[ijk-1      +kStride]-x_n[ijk+kStride]-x_n[ijk-1      -kStride]+x_n[ijk-kStride])
          );
          flux_j[flux_ij] = -bh2invSTENCIL_TWELFTH*(
                 (beta_j[ijk        ]) * ( 15.0*(x_n[ijk-jStride]-x_n[ijk]) - (x_n[ijk-jStride-jStride]-x_n[ijk+jStride]) ) + 
            0.25*(beta_j[ijk+1      ]-beta_j[ijk-1      ]) * (x_n[ijk-jStride+1      ]-x_n[ijk+1      ]-x_n[ijk-jStride-1      ]+x_n[ijk-1      ]) + 
            0.25*(beta_j[ijk+kStride]-beta_j[ijk-kStride]) * (x_n[ijk-jStride+kStride]-x_n[ijk+kStride]-x_n[ijk-jStride-kStride]+x_n[ijk-kStride])
          );
	  if(i == 0){
	  ijk = (i+idim) + j*jStride + k*kStride;
          flux_ij = (i+idim) + j*flux_jStride;
          flux_i[flux_ij] = -bh2invSTENCIL_TWELFTH*(
                 (beta_i[ijk        ]) * ( 15.0*(x_n[ijk-1      ]-x_n[ijk]) - (x_n[ijk-2              ]-x_n[ijk+1      ]) ) +
            0.25*(beta_i[ijk+jStride]-beta_i[ijk-jStride]) * (x_n[ijk-1      +jStride]-x_n[ijk+jStride]-x_n[ijk-1      -jStride]+x_n[ijk-jStride]) +
            0.25*(beta_i[ijk+kStride]-beta_i[ijk-kStride]) * (x_n[ijk-1      +kStride]-x_n[ijk+kStride]-x_n[ijk-1      -kStride]+x_n[ijk-kStride])
          );
          flux_j[flux_ij] = -bh2invSTENCIL_TWELFTH*(
                 (beta_j[ijk        ]) * ( 15.0*(x_n[ijk-jStride]-x_n[ijk]) - (x_n[ijk-jStride-jStride]-x_n[ijk+jStride]) ) +
            0.25*(beta_j[ijk+1      ]-beta_j[ijk-1      ]) * (x_n[ijk-jStride+1      ]-x_n[ijk+1      ]-x_n[ijk-jStride-1      ]+x_n[ijk-1      ]) +
            0.25*(beta_j[ijk+kStride]-beta_j[ijk-kStride]) * (x_n[ijk-jStride+kStride]-x_n[ijk+kStride]-x_n[ijk-jStride-kStride]+x_n[ijk-kStride])
          );
	  }
	  if(j == 0){
          ijk = i + (j+jdim)*jStride + k*kStride;
          flux_ij = i + (j+jdim)*flux_jStride;
          flux_i[flux_ij] = -bh2invSTENCIL_TWELFTH*(
                 (beta_i[ijk        ]) * ( 15.0*(x_n[ijk-1      ]-x_n[ijk]) - (x_n[ijk-2              ]-x_n[ijk+1      ]) ) +
            0.25*(beta_i[ijk+jStride]-beta_i[ijk-jStride]) * (x_n[ijk-1      +jStride]-x_n[ijk+jStride]-x_n[ijk-1      -jStride]+x_n[ijk-jStride]) +
            0.25*(beta_i[ijk+kStride]-beta_i[ijk-kStride]) * (x_n[ijk-1      +kStride]-x_n[ijk+kStride]-x_n[ijk-1      -kStride]+x_n[ijk-kStride])
          );
          flux_j[flux_ij] = -bh2invSTENCIL_TWELFTH*(
                 (beta_j[ijk        ]) * ( 15.0*(x_n[ijk-jStride]-x_n[ijk]) - (x_n[ijk-jStride-jStride]-x_n[ijk+jStride]) ) +
            0.25*(beta_j[ijk+1      ]-beta_j[ijk-1      ]) * (x_n[ijk-jStride+1      ]-x_n[ijk+1      ]-x_n[ijk-jStride-1      ]+x_n[ijk-1      ]) +
            0.25*(beta_j[ijk+kStride]-beta_j[ijk-kStride]) * (x_n[ijk-jStride+kStride]-x_n[ijk+kStride]-x_n[ijk-jStride-kStride]+x_n[ijk-kStride])
          );
	  }
	}
	__syncthreads();
	if(bounds){
        // calculate flux_k for k==0 once (startup / prologue)
        if(k==0){
              ijk = i + j*jStride;
          flux_ij = i + j*flux_jStride;
          flux_klo[flux_ij] = -bh2invSTENCIL_TWELFTH*(
                 (beta_k[ijk        ]) * ( 15.0*(x_n[ijk-kStride]-x_n[ijk]) - (x_n[ijk-kStride-kStride]-x_n[ijk+kStride]) ) + 
            0.25*(beta_k[ijk+1      ]-beta_k[ijk-1      ]) * (x_n[ijk-kStride+1      ]-x_n[ijk+1      ]-x_n[ijk-kStride-1      ]+x_n[ijk-1      ]) + 
            0.25*(beta_k[ijk+jStride]-beta_k[ijk-jStride]) * (x_n[ijk-kStride+jStride]-x_n[ijk+jStride]-x_n[ijk-kStride-jStride]+x_n[ijk-jStride])
          );
	}

        // calculate high flux_k on each iteration of k (becomes low on next iteration)
              ijk = i + j*jStride + (k+1)*kStride;
          flux_ij = i + j*flux_jStride;
          flux_khi[flux_ij] = -bh2invSTENCIL_TWELFTH*(
                 (beta_k[ijk        ]) * ( 15.0*(x_n[ijk-kStride]-x_n[ijk]) - (x_n[ijk-kStride-kStride]-x_n[ijk+kStride]) ) + 
            0.25*(beta_k[ijk+1      ]-beta_k[ijk-1      ]) * (x_n[ijk-kStride+1      ]-x_n[ijk+1      ]-x_n[ijk-kStride-1      ]+x_n[ijk-1      ]) + 
            0.25*(beta_k[ijk+jStride]-beta_k[ijk-jStride]) * (x_n[ijk-kStride+jStride]-x_n[ijk+jStride]-x_n[ijk-kStride-jStride]+x_n[ijk-jStride])
          );

        // CHEBYSHEV smoother...
              ijk = i + j*jStride + k*kStride;
          flux_ij = i + j*flux_jStride;
          double Ax = + flux_i[  flux_ij] - flux_i[  flux_ij+      1]
                      + flux_j[  flux_ij] - flux_j[  flux_ij+flux_jStride]
                      + flux_klo[flux_ij] - flux_khi[flux_ij        ];
          #ifdef USE_HELMHOLTZ
          Ax += a*alpha[ijk]*x_n[ijk];
          #endif
          double lambda =     Dinv_ijk();
	  x_np1[ijk] = x_n[ijk] + c1*(x_n[ijk]-x_nm1[ijk]) + c2*lambda*(rhs[ijk]-Ax);
	}
	__syncthreads();
      } // kdim
}

//------------------------------------------------------------------------------------------------------------------------------
#define STENCIL_KERNEL(log_dim_i, block_i, block_j, block_k) \
  chebyshev_kernel<log_dim_i, block_i, block_j, block_k><<<num_blocks, dim3(block_i,block_j), (block_i+1)*(BLOCKCOPY_TILE_J+1)*(4)*sizeof(double)>>>(level, x_id, rhs_id, a, b, s, chebyshev_c1, chebyshev_c2);

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
