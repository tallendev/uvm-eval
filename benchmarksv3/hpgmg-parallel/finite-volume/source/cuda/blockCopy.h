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
#define COPY_THREAD_BLOCK_SIZE			64
#define INCREMENT_THREAD_BLOCK_SIZE		64

#define READ(i)	__ldg(&read[i])

template<int log_dim, int block_type>
__global__ void copy_block_kernel(level_type level, int id, communicator_type exchange_ghosts)
{
  // one CUDA thread block operates on one HPGMG tile/block
  blockCopy_type block = exchange_ghosts.blocks[block_type][blockIdx.x];

  // copy 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block.dim.i;
  int   dim_j       = block.dim.j;
  int   dim_k       = block.dim.k;

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
    
  int  read_box = block.read.box;
  int write_box = block.write.box;
  if(read_box >=0) 
     read = level.my_boxes[ read_box].vectors[id] + level.my_boxes[ read_box].ghosts*(1+level.my_boxes[ read_box].jStride+level.my_boxes[ read_box].kStride);
  if(write_box>=0)
    write = level.my_boxes[write_box].vectors[id] + level.my_boxes[write_box].ghosts*(1+level.my_boxes[write_box].jStride+level.my_boxes[write_box].kStride);

  for(int gid=threadIdx.x;gid<dim_i*dim_j*dim_k;gid+=blockDim.x){
    // simple linear mapping of 1D threads to 3D indices
    int k=(gid/dim_i)/dim_j;
    int j=(gid/dim_i)%dim_j;
    int i=gid%dim_i;

    int  read_ijk = (i+ read_i) + (j+ read_j)* read_jStride + (k+ read_k)* read_kStride;
    int write_ijk = (i+write_i) + (j+write_j)*write_jStride + (k+write_k)*write_kStride;
    write[write_ijk] = READ(read_ijk);
  }
}

template<int log_dim, int block_type>
__global__ void increment_block_kernel(level_type level, int id, double prescale, communicator_type exchange_ghosts)
{
  // one CUDA thread block operates on one HPGMG tile/block
  blockCopy_type block = exchange_ghosts.blocks[block_type][blockIdx.x];

  // copy 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block.dim.i;
  int   dim_j       = block.dim.j;
  int   dim_k       = block.dim.k;

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
     read = level.my_boxes[ block.read.box].vectors[id] + level.my_boxes[ block.read.box].ghosts*(1+level.my_boxes[ block.read.box].jStride+level.my_boxes[ block.read.box].kStride);
     read_jStride = level.my_boxes[block.read.box ].jStride;
     read_kStride = level.my_boxes[block.read.box ].kStride;
  }
  if(block.write.box>=0){
    write = level.my_boxes[block.write.box].vectors[id] + level.my_boxes[block.write.box].ghosts*(1+level.my_boxes[block.write.box].jStride+level.my_boxes[block.write.box].kStride);
    write_jStride = level.my_boxes[block.write.box].jStride;
    write_kStride = level.my_boxes[block.write.box].kStride;
  }

  for(int gid=threadIdx.x;gid<dim_i*dim_j*dim_k;gid+=blockDim.x){
    // simple linear mapping of 1D threads to 3D indices
    int k=(gid/dim_i)/dim_j;
    int j=(gid/dim_i)%dim_j;
    int i=gid%dim_i;

    int  read_ijk = (i+ read_i) + (j+ read_j)* read_jStride + (k+ read_k)* read_kStride;
    int write_ijk = (i+write_i) + (j+write_j)*write_jStride + (k+write_k)*write_kStride;
    write[write_ijk] = prescale*write[write_ijk] + READ(read_ijk);
  }
}
#undef  KERNEL
#define KERNEL(log_dim, block_type) \
  copy_block_kernel<log_dim,block_type><<<grid,block>>>(level,id,exchange_ghosts);

extern "C"
void cuda_copy_block(level_type level, int id, communicator_type exchange_ghosts, int block_type)
{
  int block = COPY_THREAD_BLOCK_SIZE;
  int grid = exchange_ghosts.num_blocks[block_type]; if(grid<=0) return;

  int log_dim = (int)log2((double)level.dim.i);
  switch(block_type){
    case 0: KERNEL_LEVEL(log_dim,0); CUDA_ERROR break;
    case 1: KERNEL_LEVEL(log_dim,1); CUDA_ERROR break;
    case 2: KERNEL_LEVEL(log_dim,2); CUDA_ERROR break;
    default: printf("CUDA ERROR: incorrect block type, %i\n", block_type);
  }
}

#undef  KERNEL
#define KERNEL(log_dim, block_type) \
  increment_block_kernel<log_dim,block_type><<<grid,block>>>(level,id,prescale,exchange_ghosts);

extern "C"
void cuda_increment_block(level_type level, int id, double prescale, communicator_type exchange_ghosts, int block_type)
{
  int block = INCREMENT_THREAD_BLOCK_SIZE;
  int grid = exchange_ghosts.num_blocks[block_type]; if(grid<=0) return;

  int log_dim = (int)log2((double)level.dim.i);
  switch(block_type){
    case 0: KERNEL_LEVEL(log_dim,0); CUDA_ERROR break;
    case 1: KERNEL_LEVEL(log_dim,1); CUDA_ERROR break;
    case 2: KERNEL_LEVEL(log_dim,2); CUDA_ERROR break;
    default: printf("CUDA ERROR: incorrect block type, %i\n", block_type);
  }
}
