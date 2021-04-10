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

template<int log_dim, int block_type, int restrictionType>
__global__ void restriction_kernel(level_type level_c, int id_c, level_type level_f, int id_f, communicator_type restriction)
{
  // load current block
  blockCopy_type block = restriction.blocks[block_type][blockIdx.z];

  // restrict 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block.dim.i; // calculate the dimensions of the resultant coarse block
  int   dim_j       = block.dim.j;
  int   dim_k       = block.dim.k;

  // thread exit conditions
  int i = (blockIdx.x*blockDim.x + threadIdx.x);
  int j = (blockIdx.y*blockDim.y + threadIdx.y);
  if(i>=dim_i || j>=dim_j) return;

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
     read = level_f.my_boxes[ block.read.box].vectors[id_f] + level_f.my_boxes[ block.read.box].ghosts*(1+level_f.my_boxes[ block.read.box].jStride+level_f.my_boxes[ block.read.box].kStride);
     read_jStride = level_f.my_boxes[block.read.box ].jStride;
     read_kStride = level_f.my_boxes[block.read.box ].kStride;
  }
  if(block.write.box>=0){
    write = level_c.my_boxes[block.write.box].vectors[id_c] + level_c.my_boxes[block.write.box].ghosts*(1+level_c.my_boxes[block.write.box].jStride+level_c.my_boxes[block.write.box].kStride);
    write_jStride = level_c.my_boxes[block.write.box].jStride;
    write_kStride = level_c.my_boxes[block.write.box].kStride;
  }

  switch(restrictionType){
    case RESTRICT_CELL:
         for(int k=0;k<dim_k;k++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
	   double r11 = READ(read_ijk                            );
	   double r12 = READ(read_ijk+1                          );
	   double r21 = READ(read_ijk  +read_jStride             );
	   double r22 = READ(read_ijk+1+read_jStride             );
	   double r31 = READ(read_ijk               +read_kStride);
	   double r32 = READ(read_ijk+1             +read_kStride);
	   double r41 = READ(read_ijk  +read_jStride+read_kStride);
	   double r42 = READ(read_ijk+1+read_jStride+read_kStride);
	   write[write_ijk] = ( r11+r12 + r21+r22 + r31+r32 + r41+r42 ) * 0.125;
           /*write[write_ijk] = ( READ(read_ijk                            )+READ(read_ijk+1                          ) +
                                READ(read_ijk  +read_jStride             )+READ(read_ijk+1+read_jStride             ) +
                                READ(read_ijk               +read_kStride)+READ(read_ijk+1             +read_kStride) +
                                READ(read_ijk  +read_jStride+read_kStride)+READ(read_ijk+1+read_jStride+read_kStride) ) * 0.125;*/
         }break;
    case RESTRICT_FACE_I:
	 for(int k=0;k<dim_k;k++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
	   double r1 = READ(read_ijk                          );
	   double r2 = READ(read_ijk+read_jStride             );
	   double r3 = READ(read_ijk             +read_kStride);
	   double r4 = READ(read_ijk+read_jStride+read_kStride);
           write[write_ijk] = ( r1 + r2 + r3 + r4 ) * 0.25;
           /*write[write_ijk] = ( READ(read_ijk                          ) +
                                READ(read_ijk+read_jStride             ) +
                                READ(read_ijk             +read_kStride) +
                                READ(read_ijk+read_jStride+read_kStride) ) * 0.25;*/
         }break;
    case RESTRICT_FACE_J:
	 for(int k=0;k<dim_k;k++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
  	   double r1 = READ(read_ijk               );
	   double r2 = READ(read_ijk+1             );
	   double r3 = READ(read_ijk  +read_kStride);
	   double r4 = READ(read_ijk+1+read_kStride);
	   write[write_ijk] = ( r1 + r2 + r3 + r4 ) * 0.25;
           /*write[write_ijk] = ( READ(read_ijk               ) +
                                READ(read_ijk+1             ) +
                                READ(read_ijk  +read_kStride) +
                                READ(read_ijk+1+read_kStride) ) * 0.25;*/
         }break;
    case RESTRICT_FACE_K:
	 for(int k=0;k<dim_k;k++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
	   double r1 = READ(read_ijk               );
	   double r2 = READ(read_ijk+1             );
	   double r3 = READ(read_ijk  +read_jStride);
	   double r4 = READ(read_ijk+1+read_jStride);
	   write[write_ijk] = ( r1 + r2 + r3 + r4 ) * 0.25;
           /*write[write_ijk] = ( READ(read_ijk               ) +
                                READ(read_ijk+1             ) +
                                READ(read_ijk  +read_jStride) +
                                READ(read_ijk+1+read_jStride) ) * 0.25;*/
         }break;
  }
}
#undef  KERNEL
#define KERNEL(log_dim, block_type) \
  switch(restrictionType){ \
  case RESTRICT_CELL:   restriction_kernel<log_dim,block_type,RESTRICT_CELL  ><<<grid,block>>>(level_c,id_c,level_f,id_f,restriction); CUDA_ERROR break; \
  case RESTRICT_FACE_I: restriction_kernel<log_dim,block_type,RESTRICT_FACE_I><<<grid,block>>>(level_c,id_c,level_f,id_f,restriction); CUDA_ERROR break; \
  case RESTRICT_FACE_J: restriction_kernel<log_dim,block_type,RESTRICT_FACE_J><<<grid,block>>>(level_c,id_c,level_f,id_f,restriction); CUDA_ERROR break; \
  case RESTRICT_FACE_K: restriction_kernel<log_dim,block_type,RESTRICT_FACE_K><<<grid,block>>>(level_c,id_c,level_f,id_f,restriction); CUDA_ERROR break; \
  default: printf("CUDA ERROR: incorrect restriction type, %i\n", restrictionType);}

extern "C"
void cuda_restriction(level_type level_c, int id_c, level_type level_f, int id_f, communicator_type restriction, int restrictionType, int block_type)
{
  int num_blocks = restriction.num_blocks[block_type]; if(num_blocks<=0) return;
  dim3 block = dim3(min(level_c.box_dim,BLOCKCOPY_TILE_I), BLOCKCOPY_TILE_J, 1);
  dim3 grid = dim3((BLOCKCOPY_TILE_I+block.x-1)/block.x,(BLOCKCOPY_TILE_J+block.y-1)/block.y,num_blocks);

  int log_dim = (int)log2((double)level_c.dim.i);
  switch(block_type){
    case 0: KERNEL_LEVEL(log_dim, 0); break;
    case 1: KERNEL_LEVEL(log_dim, 1); break;
    default: printf("CUDA ERROR: incorrect block type, %i\n", block_type);
  }
}
