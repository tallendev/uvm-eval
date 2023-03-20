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

#define READ(i)	__ldg(&read[i])

template<int log_dim, int block_type>
__global__ void interpolation_p0_kernel(level_type level_f, int id_f, double prescale_f, level_type level_c, int id_c, communicator_type interpolation)
{
  // one CUDA thread block operates on one HPGMG tile/block
  blockCopy_type block = interpolation.blocks[block_type][blockIdx.z];

  // interpolate 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block.dim.i<<1; // calculate the dimensions of the resultant fine block
  int   dim_j       = block.dim.j<<1;
  int   dim_k       = block.dim.k<<1;

  // thread exit conditions
  int i = (blockIdx.x*blockDim.x + threadIdx.x);
  int j = (blockIdx.y*blockDim.y + threadIdx.y)*2;
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
     read = level_c.my_boxes[ block.read.box].vectors[id_c] + level_c.my_boxes[ block.read.box].ghosts*(1+level_c.my_boxes[ block.read.box].jStride+level_c.my_boxes[ block.read.box].kStride);
     read_jStride = level_c.my_boxes[block.read.box ].jStride;
     read_kStride = level_c.my_boxes[block.read.box ].kStride;
  }
  if(block.write.box>=0){
    write = level_f.my_boxes[block.write.box].vectors[id_f] + level_f.my_boxes[block.write.box].ghosts*(1+level_f.my_boxes[block.write.box].jStride+level_f.my_boxes[block.write.box].kStride);
    write_jStride = level_f.my_boxes[block.write.box].jStride;
    write_kStride = level_f.my_boxes[block.write.box].kStride;
  }

  for(int k=0;k<dim_k;k+=2){
    int write_ijk = ((i  )+write_i) + (((j  )+write_j)*write_jStride) + (((k  )+write_k)*write_kStride);
    int  read_ijk = ((i>>1)+ read_i) + (((j>>1)+ read_j)* read_jStride) + (((k>>1)+ read_k)* read_kStride);

    double rval = READ(read_ijk);

    write[write_ijk] = prescale_f*write[write_ijk] + rval;
    
    write_ijk = ((i  )+write_i) + (((j  )+write_j)*write_jStride) + (((k+1)+write_k)*write_kStride);
    write[write_ijk] = prescale_f*write[write_ijk] + rval;

    write_ijk = ((i  )+write_i) + (((j+1)+write_j)*write_jStride) + (((k  )+write_k)*write_kStride);
    write[write_ijk] = prescale_f*write[write_ijk] + rval;

    write_ijk = ((i  )+write_i) + (((j+1)+write_j)*write_jStride) + (((k+1)+write_k)*write_kStride);
    write[write_ijk] = prescale_f*write[write_ijk] + rval;
  }
}

template<int log_dim, int block_type>
__global__ void interpolation_p1_kernel(level_type level_f, int id_f, double prescale_f, level_type level_c, int id_c, communicator_type interpolation)
{
  // one CUDA thread block operates on one HPGMG tile/block
  blockCopy_type block = interpolation.blocks[block_type][blockIdx.z];

  // interpolate 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block.dim.i<<1; // calculate the dimensions of the resultant fine block
  int   dim_j       = block.dim.j<<1;
  int   dim_k       = block.dim.k<<1;

  // thread exit conditions
  int i = (blockIdx.x*blockDim.x + threadIdx.x);
  int j = (blockIdx.y*blockDim.y + threadIdx.y)*2;
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
     read = level_c.my_boxes[ block.read.box].vectors[id_c] + level_c.my_boxes[ block.read.box].ghosts*(1+level_c.my_boxes[ block.read.box].jStride+level_c.my_boxes[ block.read.box].kStride);
     read_jStride = level_c.my_boxes[block.read.box ].jStride;
     read_kStride = level_c.my_boxes[block.read.box ].kStride;
  }
  if(block.write.box>=0){
    write = level_f.my_boxes[block.write.box].vectors[id_f] + level_f.my_boxes[block.write.box].ghosts*(1+level_f.my_boxes[block.write.box].jStride+level_f.my_boxes[block.write.box].kStride);
    write_jStride = level_f.my_boxes[block.write.box].jStride;
    write_kStride = level_f.my_boxes[block.write.box].kStride;
  }

  for(int k=0;k<dim_k;k+=2){
    int write_ijk = ((i   )+write_i) + (((j   )+write_j)*write_jStride) + (((k   )+write_k)*write_kStride);
    int  read_ijk = ((i>>1)+ read_i) + (((j>>1)+ read_j)* read_jStride) + (((k>>1)+ read_k)* read_kStride);
    // linear
    int delta_i=           -1;if(i&0x1)delta_i=           1;
    int delta_j=-read_jStride;//if(j&0x1)delta_j=read_jStride;
    int delta_k=-read_kStride;//if(k&0x1)delta_k=read_kStride;

    double r1 = READ(read_ijk                        );
    double r2 = READ(read_ijk                +delta_k);
    double r3 = READ(read_ijk        +delta_j        );
    double r4 = READ(read_ijk        +delta_j+delta_k);
    double r5 = READ(read_ijk+delta_i                );
    double r6 = READ(read_ijk+delta_i        +delta_k);
    double r7 = READ(read_ijk+delta_i+delta_j        );
    double r8 = READ(read_ijk+delta_i+delta_j+delta_k);

    // i  j  k
    write[write_ijk] = prescale_f*write[write_ijk] + 0.421875*r1 + 0.140625*r2 + 0.140625*r3 + 0.046875*r4 + 0.140625*r5 + 0.046875*r6 + 0.046875*r7 + 0.015625*r8;

    // i  j  k+1
    write_ijk = ((i  )+write_i) + (((j  )+write_j)*write_jStride) + (((k+1)+write_k)*write_kStride);
    delta_j=-read_jStride;
    delta_k=read_kStride;
    double r2k = READ(read_ijk                +delta_k);
            r4 = READ(read_ijk        +delta_j+delta_k);
    double r6k = READ(read_ijk+delta_i        +delta_k);
            r8 = READ(read_ijk+delta_i+delta_j+delta_k);
    write[write_ijk] = prescale_f*write[write_ijk] + 0.421875*r1 + 0.140625*r2k+ 0.140625*r3 + 0.046875*r4 + 0.140625*r5 + 0.046875*r6k+ 0.046875*r7 + 0.015625*r8;

    // i  j+1  k
    write_ijk = ((i  )+write_i) + (((j+1)+write_j)*write_jStride) + (((k  )+write_k)*write_kStride);
    delta_j=read_jStride;
    delta_k=-read_kStride;
    double r3j = READ(read_ijk        +delta_j        );
            r4 = READ(read_ijk        +delta_j+delta_k);
    double r7j = READ(read_ijk+delta_i+delta_j        );
            r8 = READ(read_ijk+delta_i+delta_j+delta_k);
    write[write_ijk] = prescale_f*write[write_ijk] + 0.421875*r1 + 0.140625*r2 + 0.140625*r3j+ 0.046875*r4 + 0.140625*r5 + 0.046875*r6 + 0.046875*r7j+ 0.015625*r8;

    // i  j+1  k+1
    write_ijk = ((i  )+write_i) + (((j+1)+write_j)*write_jStride) + (((k+1)+write_k)*write_kStride);
    delta_j=read_jStride;
    delta_k=read_kStride;
            r4 = READ(read_ijk        +delta_j+delta_k);
            r8 = READ(read_ijk+delta_i+delta_j+delta_k);
    write[write_ijk] = prescale_f*write[write_ijk] + 0.421875*r1 + 0.140625*r2k+ 0.140625*r3j+ 0.046875*r4 + 0.140625*r5 + 0.046875*r6k+ 0.046875*r7j+ 0.015625*r8;
  }
}
#undef  KERNEL
#define KERNEL(log_dim, block_type) \
  interpolation_p0_kernel<log_dim,block_type><<<grid,block>>>(level_f,id_f,prescale_f,level_c,id_c,interpolation);

extern "C"
void cuda_interpolation_p0(level_type level_f, int id_f, double prescale_f, level_type level_c, int id_c, communicator_type interpolation, int block_type)
{
  int num_blocks = interpolation.num_blocks[block_type]; if(num_blocks<=0) return;
  dim3 block = dim3(min(level_f.box_dim,BLOCKCOPY_TILE_I), BLOCKCOPY_TILE_J, 1);
  dim3 grid = dim3((level_f.box_dim+block.x-1)/block.x, 1, num_blocks);

  int log_dim = (int)log2((double)level_f.dim.i);
  switch(block_type){
    case 0: KERNEL_LEVEL(log_dim,0); CUDA_ERROR break;
    case 1: KERNEL_LEVEL(log_dim,1); CUDA_ERROR break;
    default: printf("CUDA ERROR: incorrect block type, %i\n", block_type);
  }
}

#undef  KERNEL
#define KERNEL(log_dim, block_type) \
  interpolation_p1_kernel<log_dim,block_type><<<grid,block>>>(level_f,id_f,prescale_f,level_c,id_c,interpolation);

extern "C"
void cuda_interpolation_pl(level_type level_f, int id_f, double prescale_f, level_type level_c, int id_c, communicator_type interpolation, int block_type)
{
  int num_blocks = interpolation.num_blocks[block_type]; if(num_blocks<=0) return;
  dim3 block = dim3(min(level_f.box_dim,BLOCKCOPY_TILE_I), BLOCKCOPY_TILE_J, 1);
  dim3 grid = dim3((level_f.box_dim+block.x-1)/block.x, 1, num_blocks);

  int log_dim = (int)log2((double)level_f.dim.i);
  switch(block_type){
    case 0: KERNEL_LEVEL(log_dim,0); CUDA_ERROR break;
    case 1: KERNEL_LEVEL(log_dim,1); CUDA_ERROR break;
    default: printf("CUDA ERROR: incorrect block type, %i\n", block_type);
  }
}
