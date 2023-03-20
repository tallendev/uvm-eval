#include "chunk.h"
#include <cuda_runtime.h>
#include <map>


size_t UM_TOT_MEM = 0;
size_t UM_MAX_MEM = 0;
std::map<void*, size_t> MEMMAP;

// Initialise the chunk
void initialise_chunk(Chunk* chunk, Settings* settings, int x, int y)
{
  // Initialise the key variables
  int depth = settings->halo_depth*2;
  chunk->x = x + depth;
  chunk->y = y + depth;
  chunk->dt_init = settings->dt_init;

  // Allocate the neighbour list
  cudaMallocManaged(&chunk->neighbours, (sizeof(int) * NUM_FACES));
  // Allocate the MPI comm buffers
  int lr_len = chunk->y*settings->halo_depth*NUM_FIELDS;
  cudaMallocManaged(&chunk->left_send, sizeof(double)*lr_len);
  cudaMallocManaged(&chunk->left_recv, sizeof(double)*lr_len);
  cudaMallocManaged(&chunk->right_send, sizeof(double)*lr_len);
  cudaMallocManaged(&chunk->right_recv, sizeof(double)*lr_len);

  int tb_len = chunk->x*settings->halo_depth*NUM_FIELDS;
  cudaMallocManaged(&chunk->top_send, sizeof(double)*tb_len);
  cudaMallocManaged(&chunk->top_recv, sizeof(double)*tb_len);
  cudaMallocManaged(&chunk->bottom_send, sizeof(double)*tb_len);
  cudaMallocManaged(&chunk->bottom_recv, sizeof(double)*tb_len);

UM_TOT_MEM += sizeof(int) * NUM_FACES + 4*sizeof(double)*lr_len + sizeof(double) * tb_len * 4 + sizeof(ChunkExtension);
MEMMAP[chunk->neighbours] = sizeof(int) * NUM_FACES;

MEMMAP[chunk->left_send] = sizeof(double) * lr_len;
MEMMAP[chunk->left_recv] = sizeof(double) * lr_len;
MEMMAP[chunk->right_send] = sizeof(double) * lr_len;
MEMMAP[chunk->right_recv] = sizeof(double) * lr_len;

MEMMAP[chunk->top_send] = sizeof(double) * tb_len;
MEMMAP[chunk->top_recv] = sizeof(double) * tb_len;
MEMMAP[chunk->bottom_send] = sizeof(double) * tb_len;
MEMMAP[chunk->bottom_recv] = sizeof(double) * tb_len;


  // Initialise the ChunkExtension, which allows composition of extended
  // fields specific to individual implementations
  cudaMallocManaged(&chunk->ext, sizeof(ChunkExtension));
MEMMAP[chunk->ext] = sizeof(ChunkExtension);

UM_MAX_MEM = UM_TOT_MEM > UM_MAX_MEM ? UM_TOT_MEM : UM_MAX_MEM;


/*
  // Allocate the neighbour list
  chunk->neighbours = (int*)malloc(sizeof(int)*NUM_FACES);

  // Allocate the MPI comm buffers
  int lr_len = chunk->y*settings->halo_depth*NUM_FIELDS;
  chunk->left_send = (double*)malloc(sizeof(double)*lr_len);
  chunk->left_recv = (double*)malloc(sizeof(double)*lr_len);
  chunk->right_send = (double*)malloc(sizeof(double)*lr_len);
  chunk->right_recv = (double*)malloc(sizeof(double)*lr_len);

  int tb_len = chunk->x*settings->halo_depth*NUM_FIELDS;
  chunk->top_send = (double*)malloc(sizeof(double)*tb_len);
  chunk->top_recv = (double*)malloc(sizeof(double)*tb_len);
  chunk->bottom_send = (double*)malloc(sizeof(double)*tb_len);
  chunk->bottom_recv = (double*)malloc(sizeof(double)*tb_len);

  // Initialise the ChunkExtension, which allows composition of extended
  // fields specific to individual implementations
  chunk->ext = (ChunkExtension*)malloc(sizeof(ChunkExtension));
*/
}

void cudaFreeF(void* ptr)
{
    UM_TOT_MEM -= MEMMAP[ptr];
    MEMMAP.erase(ptr);
    cudaFree(ptr);
}

// Finalise the chunk
void finalise_chunk(Chunk* chunk)
{
  cudaFreeF(chunk->neighbours);
  cudaFreeF(chunk->ext);
  cudaFreeF(chunk->left_send);
  cudaFreeF(chunk->left_recv);
  cudaFreeF(chunk->right_send);
  cudaFreeF(chunk->right_recv);
  cudaFreeF(chunk->top_send);
  cudaFreeF(chunk->top_recv);
  cudaFreeF(chunk->bottom_send);
  cudaFreeF(chunk->bottom_recv);
}
