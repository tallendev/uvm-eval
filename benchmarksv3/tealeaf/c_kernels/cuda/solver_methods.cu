#include "cuknl_shared.h"
#include "../../shared.h"

__global__ void sum_reduce(
        const int n, double* buffer);

void sum_reduce_buffer(
        double* buffer, double* result, int len)
{
    while(len > 1)
    {
        int num_blocks = ceil(len / (double)BLOCK_SIZE);
        sum_reduce<<<num_blocks, BLOCK_SIZE>>>(len, buffer);
        len = num_blocks;
    }

    check_errors(__LINE__, __FILE__);
    //cudaMemcpy(result, buffer, sizeof(double), cudaMemcpyDeviceToHost);
    *result = *buffer;
}

__global__ void copy_u(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double* src,
        double* dest)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner; 
    const int off0 = halo_depth*(x + 1);
    const int index = off0 + col + row*x;

    dest[index] = src[index];	
}

__global__ void calculate_residual(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double* u,
        const double* u0,
        const double* kx,
        const double* ky,
        double* r)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner; 
    const int off0 = halo_depth*(x + 1);
    const int index = off0 + col + row*x;

    const double smvp = SMVP(u);
    r[index] = u0[index] - smvp;
}

__global__ void calculate_2norm(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double* src,
        double* norm)
{
    __shared__ double norm_shared[BLOCK_SIZE];
    norm_shared[threadIdx.x] = 0.0;

    const int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner; 
    const int off0 = halo_depth*(x + 1);
    const int index = off0 + col + row*x;

    norm_shared[threadIdx.x] = src[index]*src[index];

    reduce<double, BLOCK_SIZE/2>::run(norm_shared, norm, SUM);
}

__global__ void finalise(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double* density,
        const double* u,
        double* energy)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner; 
    const int off0 = halo_depth*(x + 1);
    const int index = off0 + col + row*x;

    energy[index] = u[index]/density[index];
}

__global__ void sum_reduce(
        const int n,
        double* buffer)
{
    __shared__ double buffer_shared[BLOCK_SIZE];

    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    buffer_shared[threadIdx.x] = (gid < n) ? buffer[gid] : 0.0;

    reduce<double, BLOCK_SIZE/2>::run(buffer_shared, buffer, SUM);
}

__global__ void zero_buffer(
        const int x,
        const int y,
        double* buffer)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid < x*y)
    {
        buffer[gid] = 0.0;
    }
}
