#pragma once

typedef double* FieldBufferType;

// Empty extension point
typedef struct ChunkExtension
{
    double* d_comm_buffer;
    double* d_reduction_buffer1;
    double* d_reduction_buffer2;
    double* d_reduction_buffer3;
    double* d_reduction_buffer4;
} ChunkExtension;
