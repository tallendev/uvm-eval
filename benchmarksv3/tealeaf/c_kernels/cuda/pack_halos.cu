#include <stdlib.h>
#include "../../chunk.h"
#include "../../shared.h"
#include "cuknl_shared.h"

typedef void (*pack_kernel_f)( 
		const int x, const int y, const int halo_depth, double* field,
		double* buffer, const int depth);

__global__ void pack_left(
		const int x, const int y, const int halo_depth, double* field,
		double* buffer, const int depth);
__global__ void pack_right(
		const int x, const int y, const int halo_depth, double* field,
		double* buffer, const int depth);
__global__ void pack_top(
		const int x, const int y, const int halo_depth, double* field,
		double* buffer, const int depth);
__global__ void pack_bottom(
		const int x, const int y, const int halo_depth, double* field,
		double* buffer, const int depth);
__global__ void unpack_left(
		const int x, const int y, const int halo_depth, double* field,
		double* buffer, const int depth);
__global__ void unpack_right(
		const int x, const int y, const int halo_depth, double* field,
		double* buffer, const int depth);
__global__ void unpack_top( 
        const int x, const int y, const int halo_depth, double* field, 
        double* buffer, const int depth);
__global__ void unpack_bottom( 
        const int x, const int y, const int halo_depth, double* field, 
        double* buffer, const int depth);

// Either packs or unpacks data from/to buffers.
void pack_or_unpack(
        Chunk* chunk, Settings* settings, int depth, int face, 
        bool pack, double* field, double* buffer)
{
    pack_kernel_f kernel = NULL;

    const int x_inner = chunk->x - 2*settings->halo_depth;
    const int y_inner = chunk->y - 2*settings->halo_depth;

    int buffer_length = 0;

    switch(face)
    {
        case CHUNK_LEFT:
            kernel = pack ? pack_left : unpack_left;
            buffer_length = y_inner*depth;
            break;
        case CHUNK_RIGHT:
            kernel = pack ? pack_right : unpack_right;
            buffer_length = y_inner*depth;
            break;
        case CHUNK_TOP:
            kernel = pack ? pack_top : unpack_top;
            buffer_length = x_inner*depth;
            break;
        case CHUNK_BOTTOM:
            kernel = pack ? pack_bottom : unpack_bottom;
            buffer_length = x_inner*depth;
            break;
        default:
            die(__LINE__, __FILE__, "Incorrect face provided: %d.\n", face);
    }

    if(!pack)
    {
        //cudaMemcpy(
        //        chunk->ext->d_comm_buffer, buffer, buffer_length*sizeof(double), 
        //        cudaMemcpyHostToDevice);
        check_errors(__LINE__, __FILE__);
    }
    int num_blocks = ceil(buffer_length / (double)BLOCK_SIZE);
    kernel<<<num_blocks, BLOCK_SIZE>>>(
            chunk->x, chunk->y, settings->halo_depth, field,
            buffer, depth);
            //chunk->ext->d_comm_buffer, depth);

    if(pack)
    {
        //cudaMemcpy(
        //        buffer, chunk->ext->d_comm_buffer, buffer_length*sizeof(double),
        //        cudaMemcpyDeviceToHost);
        check_errors(__LINE__, __FILE__);
    }
}

__global__ void pack_left(
        const int x,
        const int y,
        const int halo_depth,
        double* field,
        double* buffer,
        const int depth)
{
    const int y_inner = y - 2*halo_depth;

    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= y_inner*depth) return;

    const int lines = gid / depth;
    const int offset = halo_depth + lines*(x - depth);
    buffer[gid] = field[offset+gid];
}

__global__ void pack_right(
        const int x,
        const int y,
        const int halo_depth,
        double* field,
        double* buffer,
        const int depth)
{
    const int y_inner = y - 2*halo_depth;

    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= y_inner*depth) return;

    const int lines = gid / depth;
    const int offset = x - halo_depth - depth + lines*(x - depth);
    buffer[gid] = field[offset+gid];
}

__global__ void unpack_left(
        const int x,
        const int y,
        const int halo_depth,
        double* field,
        double* buffer,
        const int depth)
{
    const int y_inner = y - 2*halo_depth;

    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= y_inner*depth) return;

    const int lines = gid / depth;
    const int offset = halo_depth - depth + lines*(x - depth);
    field[offset+gid] = buffer[gid];
}

__global__ void unpack_right(
        const int x,
        const int y,
        const int halo_depth,
        double* field,
        double* buffer,
        const int depth)
{
    const int y_inner = y - 2*halo_depth;

    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= y_inner*depth) return;

    const int lines = gid / depth;
    const int offset = x - halo_depth + lines*(x - depth);
    field[offset+gid] = buffer[gid];
}

__global__ void pack_top(
        const int x,
        const int y,
        const int halo_depth,
        double* field,
        double* buffer,
        const int depth)
{
    const int x_inner = x - 2*halo_depth;

    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= x_inner*depth) return;

    const int lines = gid / x_inner;
    const int offset = x - halo_depth + lines*(x - depth);
    buffer[gid] = field[offset+gid];
}

__global__ void pack_bottom(
        const int x,
        const int y,
        const int halo_depth,
        double* field,
        double* buffer,
        const int depth)
{
    const int x_inner = x - 2*halo_depth;

    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= x_inner*depth) return;

    const int lines = gid / x_inner;
    const int offset = x*halo_depth + lines*2*halo_depth;
    buffer[gid] = field[offset+gid];
}

__global__ void unpack_top(
        const int x,
        const int y,
        const int halo_depth,
        double* field,
        double* buffer,
        const int depth)
{
    const int x_inner = x - 2*halo_depth;

    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= x_inner*depth) return;

    const int lines = gid / x_inner;
    const int offset = x*(y - halo_depth) + lines*2*halo_depth;
    field[offset+gid] = buffer[gid];
}

__global__ void unpack_bottom(
        const int x,
        const int y,
        const int halo_depth,
        double* field,
        double* buffer,
        const int depth)
{
    const int x_inner = x - 2*halo_depth;

    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= x_inner*depth) return;

    const int lines = gid / x_inner;
    const int offset = x*(halo_depth - depth) + lines*2*halo_depth;
    field[offset+gid] = buffer[gid];
}

