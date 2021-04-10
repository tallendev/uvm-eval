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
// Nikolay Sakharnykh
// nsakharnykh@nvidia.com
// Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
//------------------------------------------------------------------------------------------------------------------------------
#ifdef _OPENMP
#include <omp.h>
#endif
//------------------------------------------------------------------------------------------------------------------------------
#include "../timers.h"
#include "../defines.h"
#include "../level.h"
#include "../operators.h"
//------------------------------------------------------------------------------------------------------------------------------
#define STENCIL_VARIABLE_COEFFICIENT
//------------------------------------------------------------------------------------------------------------------------------
#ifdef STENCIL_FUSE_BC
  #error GPU implementation does not support fusion of the boundary conditions with the operator
#endif
//------------------------------------------------------------------------------------------------------------------------------
#ifdef USE_TEX
  #define       TEX
  #define  X(i)  ( __ldg(&x[i])      )
  #define BI(i)  ( __ldg(&beta_i[i]) )
  #define BJ(i)  ( __ldg(&beta_j[i]) )
  #define BK(i)  ( __ldg(&beta_k[i]) )
#elif  USE_TEXPA
  #define       TEX
  #define  X(i)  ( __ldg(x+i)      )
  #define BI(i)  ( __ldg(beta_i+i) )
  #define BJ(i)  ( __ldg(beta_j+i) )
  #define BK(i)  ( __ldg(beta_k+i) )
#else
  #define  X(i)  ( x[i]      )
  #define BI(i)  ( beta_i[i] )
  #define BJ(i)  ( beta_j[i] )
  #define BK(i)  ( beta_k[i] )
#endif
//------------------------------------------------------------------------------------------------------------------------------
#define Dinv_ijk() Dinv[ijk]        // simply retrieve it rather than recalculating it
//------------------------------------------------------------------------------------------------------------------------------
#ifdef STENCIL_VARIABLE_COEFFICIENT
  #ifdef USE_HELMHOLTZ // Helmholtz
  #define H0	( a*alpha[ijk]*X(ijk) )
  #else // Poisson
  #define H0
  #endif
  #define apply_op_ijk()                        \
  (                                  		\
  H0 - b*h2inv*(				\
  + BI(ijk+1      )*( X(ijk+1      ) - X(ijk) ) \
  + BI(ijk        )*( X(ijk-1      ) - X(ijk) ) \
  + BJ(ijk+jStride)*( X(ijk+jStride) - X(ijk) ) \
  + BJ(ijk        )*( X(ijk-jStride) - X(ijk) ) \
  + BK(ijk+kStride)*( X(ijk+kStride) - X(ijk) ) \
  + BK(ijk        )*( X(ijk-kStride) - X(ijk) ) \
  )        				        \
  )
#else // constant coefficient
  #define apply_op_ijk()  \
  (                       \
  a*X(ijk) - b*h2inv*(    \
  + X(ijk+1      )        \
  + X(ijk-1      )        \
  + X(ijk+jStride)        \
  + X(ijk-jStride)        \
  + X(ijk+kStride)        \
  + X(ijk-kStride)        \
  - X(ijk        )*6.0    \
  )                       \
  )
#endif
//------------------------------------------------------------------------------------------------------------------------------
#ifdef  USE_GSRB
#define NUM_SMOOTHS      3 // RBRBRB
#elif   USE_CHEBY
#define NUM_SMOOTHS      1
#define CHEBYSHEV_DEGREE 6 // i.e. one degree-6 polynomial smoother
#elif   USE_JACOBI
#define NUM_SMOOTHS      6
#elif   USE_L1JACOBI
#define NUM_SMOOTHS      6
#else
#error You must compile CUDA code with either -DUSE_GSRB, -DUSE_CHEBY, -DUSE_JACOBI, -DUSE_L1JACOBI, or -DUSE_SYMGS
#endif
//------------------------------------------------------------------------------------------------------------------------------
// include smoother
#include "extra.h"
#if defined(USE_GSRB) && ( defined(GSRB_STRIDE2) || defined(GSRB_BRANCH) || (defined(GSRB_FP)&&!defined(GSRB_OOP)) )
  #include "stencils/gsrb.h"
#else
  #ifdef USE_SHM // shared memory
  #include "stencils/smooth.reg.fv2.h"
  #elif  USE_REG // registers
  #include "stencils/smooth.reg.fv2.h"
  #else // baseline
  #include "stencils/smooth.base.h"
  #endif
#endif
//------------------------------------------------------------------------------------------------------------------------------
// include residual
#ifdef USE_SHM // shared memory
#include "stencils/residual.reg.fv2.h"
#elif  USE_REG // registers
#include "stencils/residual.reg.fv2.h"
#else // baseline
#include "stencils/residual.base.h"
#endif
//------------------------------------------------------------------------------------------------------------------------------
// include other kernels
#include "blockCopy.h"
#include "misc.h"
#include "boundary_fv.h"
#include "restriction.h"
#include "interpolation_v2.h"
//------------------------------------------------------------------------------------------------------------------------------
