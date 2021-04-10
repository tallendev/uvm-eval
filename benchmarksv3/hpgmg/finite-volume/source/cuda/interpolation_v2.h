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

#undef  READ
#define READ(i)	__ldg(&read[i])

template<int log_dim, int block_type>
__global__ void interpolation_v2_kernel(level_type level_f, int id_f, double prescale_f, level_type level_c, int id_c, communicator_type interpolation){
  // one CUDA thread block operates on one HPGMG tile/block
  blockCopy_type block = interpolation.blocks[block_type][blockIdx.z];

  // interpolate 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int write_dim_i   = block.dim.i<<1; // calculate the dimensions of the resultant fine block
  int write_dim_j   = block.dim.j<<1;
  int write_dim_k   = block.dim.k<<1;

  int i = (blockIdx.x*blockDim.x + threadIdx.x);
  int j = (blockIdx.y*blockDim.y + threadIdx.y)*2;
  if(i>=write_dim_i || j>=write_dim_j) return;

  int  read_i       = block.read.i;
  int  read_j       = block.read.j;
  int  read_k       = block.read.k;
  int  read_jStride = block.read.jStride;
  int  read_kStride = block.read.kStride;

  int write_i       = block.write.i;
  int write_j       = block.write.j;
  int write_k       = block.write.k;
  int write_jStride = block.write.jStride;
  int write_kStride = block.write.kStride;

  double * __restrict__  read = block.read.ptr;
  double * __restrict__ write = block.write.ptr;
  if(block.read.box >=0){
     read_jStride = level_c.my_boxes[block.read.box ].jStride;
     read_kStride = level_c.my_boxes[block.read.box ].kStride;
     read = level_c.my_boxes[ block.read.box].vectors[id_c] + level_c.my_boxes[ block.read.box].ghosts*(1+ read_jStride+ read_kStride);
  }
  if(block.write.box>=0){
    write_jStride = level_f.my_boxes[block.write.box].jStride;
    write_kStride = level_f.my_boxes[block.write.box].kStride;
    write = level_f.my_boxes[block.write.box].vectors[id_f] + level_f.my_boxes[block.write.box].ghosts*(1+write_jStride+write_kStride);
  }

  double c1 = 1.0/8.0;
  for(int k=0;k<write_dim_k;k+=2){
    double c1i=c1;if(i&0x1){c1i=-c1;}
    double c1j=c1;//if(j&0x1){c1j=-c1;}
    double c1k=c1;//if(k&0x1){c1k=-c1;}
    int write_ijk = ((i   )+write_i) + (((j   )+write_j)*write_jStride) + (((k   )+write_k)*write_kStride);
    int  read_ijk = ((i>>1)+ read_i) + (((j>>1)+ read_j)* read_jStride) + (((k>>1)+ read_k)* read_kStride);
    //
    // |  1/8  |  1.0  | -1/8  | coarse grid
    // |---+---|---+---|---+---|
    // |   |   |???|   |   |   | fine grid
    //
    double r11 = READ(read_ijk-1-read_jStride-read_kStride);
    double r12 = READ(read_ijk-read_jStride-read_kStride  );
    double r13 = READ(read_ijk+1-read_jStride-read_kStride);
    double r21 = READ(read_ijk-1             -read_kStride);
    double r22 = READ(read_ijk             -read_kStride  );
    double r23 = READ(read_ijk+1             -read_kStride);
    double r31 = READ(read_ijk-1+read_jStride-read_kStride);
    double r32 = READ(read_ijk+read_jStride-read_kStride  );
    double r33 = READ(read_ijk+1+read_jStride-read_kStride);
    double r41 = READ(read_ijk-1-read_jStride             );
    double r42 = READ(read_ijk-read_jStride               );
    double r43 = READ(read_ijk+1-read_jStride             );
    double r51 = READ(read_ijk-1                          );
    double r52 = READ(read_ijk                            );
    double r53 = READ(read_ijk+1                          );
    double r61 = READ(read_ijk-1+read_jStride             );
    double r62 = READ(read_ijk+read_jStride               );
    double r63 = READ(read_ijk+1+read_jStride             );
    double r71 = READ(read_ijk-1-read_jStride+read_kStride);
    double r72 = READ(read_ijk-read_jStride+read_kStride  );
    double r73 = READ(read_ijk+1-read_jStride+read_kStride);
    double r81 = READ(read_ijk-1             +read_kStride);
    double r82 = READ(read_ijk             +read_kStride  );
    double r83 = READ(read_ijk+1             +read_kStride);
    double r91 = READ(read_ijk-1+read_jStride+read_kStride);
    double r92 = READ(read_ijk+read_jStride+read_kStride  );
    double r93 = READ(read_ijk+1+read_jStride+read_kStride);
 
    // i  j  k
    write[write_ijk] = prescale_f*write[write_ijk] +
                       + c1k*( + c1j*( c1i*r11 + r12 - c1i*r13 )
                               +     ( c1i*r21 + r22 - c1i*r23 )
                               - c1j*( c1i*r31 + r32 - c1i*r33 ) )
                       +     ( + c1j*( c1i*r41 + r42 - c1i*r43 )
                               +     ( c1i*r51 + r52 - c1i*r53 )
                               - c1j*( c1i*r61 + r62 - c1i*r63 ) )
                       - c1k*( + c1j*( c1i*r71 + r72 - c1i*r73 )
                               +     ( c1i*r81 + r82 - c1i*r83 )
                               - c1j*( c1i*r91 + r92 - c1i*r93 ) );

   // i  j+1  k
   write_ijk = ((i  )+write_i) + (((j+1)+write_j)*write_jStride) + (((k  )+write_k)*write_kStride);  c1j=-c1;c1k=c1;
   write[write_ijk] = prescale_f*write[write_ijk] +
                       + c1k*( + c1j*( c1i*r11 + r12 - c1i*r13 )
                               +     ( c1i*r21 + r22 - c1i*r23 )
                               - c1j*( c1i*r31 + r32 - c1i*r33 ) )
                       +     ( + c1j*( c1i*r41 + r42 - c1i*r43 )
                               +     ( c1i*r51 + r52 - c1i*r53 )
                               - c1j*( c1i*r61 + r62 - c1i*r63 ) )
                       - c1k*( + c1j*( c1i*r71 + r72 - c1i*r73 )
                               +     ( c1i*r81 + r82 - c1i*r83 )
                               - c1j*( c1i*r91 + r92 - c1i*r93 ) );

   // i  j  k+1
   write_ijk = ((i  )+write_i) + (((j  )+write_j)*write_jStride) + (((k+1)+write_k)*write_kStride);  c1j=c1;c1k=-c1;
   write[write_ijk] = prescale_f*write[write_ijk] +
                       + c1k*( + c1j*( c1i*r11 + r12 - c1i*r13 )
                               +     ( c1i*r21 + r22 - c1i*r23 )
                               - c1j*( c1i*r31 + r32 - c1i*r33 ) )
                       +     ( + c1j*( c1i*r41 + r42 - c1i*r43 )
                               +     ( c1i*r51 + r52 - c1i*r53 )
                               - c1j*( c1i*r61 + r62 - c1i*r63 ) )
                       - c1k*( + c1j*( c1i*r71 + r72 - c1i*r73 )
                               +     ( c1i*r81 + r82 - c1i*r83 )
                               - c1j*( c1i*r91 + r92 - c1i*r93 ) );

    // i  j+1  k+1
    write_ijk = ((i  )+write_i) + (((j+1)+write_j)*write_jStride) + (((k+1)+write_k)*write_kStride);  c1j=-c1;c1k=-c1;
    write[write_ijk] = prescale_f*write[write_ijk] +
                       + c1k*( + c1j*( c1i*r11 + r12 - c1i*r13 )
                               +     ( c1i*r21 + r22 - c1i*r23 )
                               - c1j*( c1i*r31 + r32 - c1i*r33 ) )
                       +     ( + c1j*( c1i*r41 + r42 - c1i*r43 )
                               +     ( c1i*r51 + r52 - c1i*r53 )
                               - c1j*( c1i*r61 + r62 - c1i*r63 ) )
                       - c1k*( + c1j*( c1i*r71 + r72 - c1i*r73 )
                               +     ( c1i*r81 + r82 - c1i*r83 )
                               - c1j*( c1i*r91 + r92 - c1i*r93 ) );
  }
}
#undef  KERNEL
#define KERNEL(log_dim, block_type) \
  interpolation_v2_kernel<log_dim,block_type><<<grid,block>>>(level_f,id_f,prescale_f,level_c,id_c,interpolation);

extern "C"
void cuda_interpolation_v2(level_type level_f, int id_f, double prescale_f, level_type level_c, int id_c, communicator_type interpolation, int block_type)
{
  int num_blocks = interpolation.num_blocks[block_type]; if(num_blocks<=0) return;
  dim3 block = dim3(min(level_f.box_dim,BLOCKCOPY_TILE_I), BLOCKCOPY_TILE_J, 1);
  dim3 grid = dim3((level_f.box_dim+block.x-1)/block.x,1,num_blocks);

  int log_dim = (int)log2((double)level_f.dim.i);
  switch(block_type){
    case 0: KERNEL_LEVEL(log_dim,0); CUDA_ERROR break;
    case 1: KERNEL_LEVEL(log_dim,1); CUDA_ERROR break;
    default: printf("CUDA ERROR: incorrect block type, %i\n", block_type);
  }
}
