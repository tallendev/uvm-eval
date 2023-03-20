#pragma once
#ifndef __CHUNKEXTENSIONH
#define __CHUNKEXTENSIONH

typedef double* FieldBufferType;

// Empty extension point
typedef struct ChunkExtension
{
    double* d_comm_buffer;
    double* d_reduce_buffer;
    double* d_reduce_buffer2;
    double* d_reduce_buffer3;
    double* d_reduce_buffer4;

} ChunkExtension;

#endif
