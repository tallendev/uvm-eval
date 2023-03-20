#include <stdlib.h>
#include "../../shared.h"

/*
 * 		LOCAL HALOS KERNEL
 */	

void update_left(const int x, const int y, 
        const int halo_depth, const int depth, double* buffer);
void update_right(const int x, const int y, 
        const int halo_depth, const int depth, double* buffer);
void update_top(const int x, const int y, 
        const int halo_depth, const int depth, double* buffer); 
void update_bottom(const int x, const int y, 
        const int halo_depth, const int depth, double* buffer);
void update_face(const int x, const int y, const int halo_depth, 
        const int* chunk_neighbours, const int depth, double* buffer);

typedef void (*update_kernel)(int,double*);

// The kernel for updating halos locally
void local_halos(
        const int x,
        const int y,
        const int depth,
        const int halo_depth,
        const int* chunk_neighbours,
        const bool* fields_to_exchange,
        double* density,
        double* energy0,
        double* energy,
        double* u,
        double* p,
        double* sd)
{
#define LAUNCH_UPDATE(index, buffer)\
    if(fields_to_exchange[index])\
    {\
        update_face(x, y, halo_depth, chunk_neighbours, depth, buffer);\
    }

    LAUNCH_UPDATE(FIELD_DENSITY, density);
    LAUNCH_UPDATE(FIELD_P, p);
    LAUNCH_UPDATE(FIELD_ENERGY0, energy0);
    LAUNCH_UPDATE(FIELD_ENERGY1, energy);
    LAUNCH_UPDATE(FIELD_U, u);
    LAUNCH_UPDATE(FIELD_SD, sd);
#undef LAUNCH_UPDATE
}

// Updates faces in turn.
void update_face(
        const int x,
        const int y, 
        const int halo_depth,
        const int* chunk_neighbours,
        const int depth,
        double* buffer)
{
#define UPDATE_FACE(face, updateKernel) \
    if(chunk_neighbours[face] == EXTERNAL_FACE)\
    {\
        updateKernel(x, y, halo_depth, depth,buffer);\
    }

    UPDATE_FACE(CHUNK_LEFT, update_left);
    UPDATE_FACE(CHUNK_RIGHT, update_right);
    UPDATE_FACE(CHUNK_TOP, update_top);
    UPDATE_FACE(CHUNK_BOTTOM, update_bottom);
}

// Update left halo.
void update_left(
        const int x,
        const int y,
        const int halo_depth,
        const int depth, 
        double* buffer)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = 0; kk < depth; ++kk)
        {
            int base = jj*x;
            buffer[base+(halo_depth-kk-1)] = buffer[base+(halo_depth+kk)];			
        }
    }
}

// Update right halo.
void update_right(
        const int x,
        const int y,
        const int halo_depth,
        const int depth,
        double* buffer)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = 0; kk < depth; ++kk)
        {
            int base = jj*x;
            buffer[base+(x-halo_depth+kk)] 
                = buffer[base+(x-halo_depth-1-kk)];
        }
    }
}

// Update top halo.
void update_top(
        const int x,
        const int y,
        const int halo_depth,
        const int depth, 
        double* buffer)
{
    for(int jj = 0; jj < depth; ++jj)
    {
#pragma omp parallel for
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            int base = kk;
            buffer[base+(y-halo_depth+jj)*x] 
                = buffer[base+(y-halo_depth-1-jj)*x];
        }
    }
}

// Updates bottom halo.
void update_bottom(
        const int x,
        const int y,
        const int halo_depth,
        const int depth, 
        double* buffer)
{
    for(int jj = 0; jj < depth; ++jj)
    {
#pragma omp parallel for
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            int base = kk;
            buffer[base+(halo_depth-jj-1)*x] 
                = buffer[base+(halo_depth+jj)*x];
        }
    }
}

