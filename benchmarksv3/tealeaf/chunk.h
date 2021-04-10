#pragma once
#ifndef __CHUNKH
#define __CHUNKH

#include <math.h>
#include "settings.h"
#include "chunk_extension.h"

// The core Tealeaf interface class.
typedef struct Chunk
{
    // Solve-wide variables
    double dt_init;

    // Neighbouring ranks
    int* neighbours; 

    // MPI comm buffers
    double* left_send;
    double* left_recv;
    double* right_send;
    double* right_recv;
    double* top_send;
    double* top_recv;
    double* bottom_send;
    double* bottom_recv;
    
    // Mesh chunks
    int left;
    int right;
    int bottom;
    int top;

    // Field dimensions
    int x;
    int y;

    // Field buffers
    FieldBufferType density0;
    FieldBufferType density;
    FieldBufferType energy0;
    FieldBufferType energy;

    FieldBufferType u;
    FieldBufferType u0;
    FieldBufferType p;
    FieldBufferType r;
    FieldBufferType mi;
    FieldBufferType w;
    FieldBufferType kx;
    FieldBufferType ky;
    FieldBufferType sd;

    FieldBufferType cell_x;
    FieldBufferType cell_y;
    FieldBufferType cell_dx;
    FieldBufferType cell_dy;

    FieldBufferType vertex_dx;
    FieldBufferType vertex_dy;
    FieldBufferType vertex_x;
    FieldBufferType vertex_y;

    FieldBufferType volume;
    FieldBufferType x_area;
    FieldBufferType y_area;

    // Cheby and PPCG  
    double theta;
    double eigmin;
    double eigmax;

    double* cg_alphas;
    double* cg_betas;
    double* cheby_alphas;
    double* cheby_betas;

    struct ChunkExtension* ext;
} Chunk;

struct Settings;

void initialise_chunk(Chunk* chunk, struct Settings* settings, int x, int y);
void finalise_chunk(Chunk* chunk);

#endif
