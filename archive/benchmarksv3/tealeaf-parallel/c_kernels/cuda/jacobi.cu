#include "../../shared.h"
#include "cuknl_shared.h"

// Core computation for Jacobi solver.
__global__ void jacobi_iterate(
		const int x_inner,
		const int y_inner,
        const int halo_depth,
		const double* kx,
		const double* ky,
		const double* u0,
		const double* r,
		double* u,
		double* error)
{
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
	__shared__ double error_local[BLOCK_SIZE];

    const int x = x_inner + 2*halo_depth;
	const int col = gid % x_inner;
	const int row = gid / x_inner; 
	const int off0 = halo_depth*(x + 1);
	const int index = off0 + col + row*x;

	if(gid < x_inner*y_inner)
	{
		u[index] = (u0[index] 
				+ kx[index+1]*r[index+1] 
				+ kx[index]*r[index-1]
				+ ky[index+x]*r[index+x] 
				+ ky[index]*r[index-x])
			/ (1.0 + (kx[index]+kx[index+1])
					+ (ky[index]+ky[index+x]));

		error_local[threadIdx.x] = fabs(u[index]-r[index]);
	}
	else
	{
		error_local[threadIdx.x] = 0.0;
	}

	reduce<double, BLOCK_SIZE/2>::run(error_local, error, SUM);
}

__global__ void jacobi_init(
		const int x_inner,
		const int y_inner,
        const int halo_depth,
		const double* density,
		const double* energy,
		const double rx,
		const double ry,
		double* kx,
		double* ky,
		double* u0,
		double* u,
		const int coefficient)
{
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
	if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner; 
    const int off0 = halo_depth*(x + 1);
    const int index = off0 + col + row*x;

    const double u_temp = energy[index]*density[index];
    u0[index] = u_temp;
    u[index] = u_temp;

    if(row == 0 || col == 0) return; 

    double density_center;
    double density_left;
    double density_down;

    if(coefficient == CONDUCTIVITY)
    {
        density_center = density[index];
        density_left = density[index-1];
        density_down = density[index-x];
    }
    else if(coefficient == RECIP_CONDUCTIVITY)
    {
        density_center = 1.0/density[index];
        density_left = 1.0/density[index-1];
        density_down = 1.0/density[index-x];
    }

    kx[index] = rx*(density_left+density_center) /
        (2.0*density_left*density_center);
    ky[index] = ry*(density_down+density_center) /
        (2.0*density_down*density_center);
}

__global__ void jacobi_copy_u(
        const int x_inner,
        const int y_inner,
        const double* src,
        double* dest)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid < x_inner*y_inner)
    {
        dest[gid] = src[gid];	
    }
}
