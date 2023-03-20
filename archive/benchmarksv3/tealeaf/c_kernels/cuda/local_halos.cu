#include <stdlib.h>
#include "../../shared.h"
#include "cuknl_shared.h"

/*
 * 		LOCAL HALOS KERNEL
 */	
__global__ void update_bottom(
		const int x, const int y, const int halo_depth, 
        const int depth, double* buffer);
__global__ void update_top(
		const int x, const int y, const int halo_depth, 
        const int depth, double* buffer);
__global__ void update_left(
		const int x, const int y, const int halo_depth, 
        const int depth, double* buffer);
__global__ void update_right(
		const int x, const int y, const int halo_depth, 
        const int depth, double* buffer);

void update_face(const int x, const int y, const int halo_depth,
        const int* chunk_neighbours, const int depth, double* buffer);

// The kernel for updating halos locally
void local_halos(
        const int x,
        const int y,
        const int halo_depth,
        const int depth,
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
#define UPDATE_FACE(face, update_kernel) \
    if(chunk_neighbours[face] == EXTERNAL_FACE) \
    {\
        update_kernel<<<num_blocks, BLOCK_SIZE>>>( \
                x, y, halo_depth, depth, buffer); \
        check_errors(__LINE__, __FILE__);\
    }

    int num_blocks = ceil((x*depth) / (double)BLOCK_SIZE);
    UPDATE_FACE(CHUNK_TOP, update_top);
    UPDATE_FACE(CHUNK_BOTTOM, update_bottom);

    num_blocks = ceil((y*depth) / (float)BLOCK_SIZE);
    UPDATE_FACE(CHUNK_RIGHT, update_right);
    UPDATE_FACE(CHUNK_LEFT, update_left);
}

__global__ void update_bottom(
        const int x,
        const int y,
        const int halo_depth,
        const int depth,
        double* buffer)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x*depth) return;

    const int lines = gid/x;
    const int offset = x*halo_depth;
    const int from_index = offset + gid;
    const int to_index = from_index - (1 + lines*2)*x;
    buffer[to_index] = buffer[from_index];
}

__global__ void update_top(
        const int x,
        const int y,
        const int halo_depth,
        const int depth,
        double* buffer)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x*depth) return;

    const int lines = gid/x;
    const int offset = x*(y - halo_depth);
    const int to_index = offset + gid;
    const int from_index = to_index - (1 + lines*2)*x;
    buffer[to_index] = buffer[from_index];
}

__global__ void update_left(
        const int x,
        const int y,
        const int halo_depth,
        const int depth,
        double* buffer)
{
    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= y*depth) return;

    const int flip = gid % depth;
    const int lines = gid / depth;
    const int offset = halo_depth + lines*(x - depth);
    const int from_index = offset + gid;
    const int to_index = from_index - (1 + flip*2);

    buffer[to_index] = buffer[from_index];
}

__global__ void update_right(
        const int x,
        const int y,
        const int halo_depth,
        const int depth,
        double* buffer)
{
    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= y*depth) return;

    const int flip = gid % depth;
    const int lines = gid / depth;
    const int offset = x - halo_depth + lines*(x - depth);
    const int to_index = offset + gid;
    const int from_index = to_index - (1 + flip*2);

    buffer[to_index] = buffer[from_index];
}

