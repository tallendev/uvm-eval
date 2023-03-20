#include "cuknl_shared.h"

__global__ void field_summary(
		const int x_inner,
		const int y_inner,
        const int halo_depth,
		const double* volume,
		const double* density,
		const double* energy0,
		const double* u,
		double* vol_out,
		double* mass_out,
		double* ie_out,
		double* temp_out)
{
	const int gid = threadIdx.x+blockDim.x*blockIdx.x;
	const int lid = threadIdx.x;

	__shared__ double vol_shared[BLOCK_SIZE];
	__shared__ double mass_shared[BLOCK_SIZE];
	__shared__ double ie_shared[BLOCK_SIZE];
	__shared__ double temp_shared[BLOCK_SIZE];

	vol_shared[lid] = 0.0;
	mass_shared[lid] = 0.0;
	ie_shared[lid] = 0.0;
	temp_shared[lid] = 0.0;

    if(gid < x_inner*y_inner)
    {
        const int x = x_inner + 2*halo_depth;
        const int col = gid % x_inner;
        const int row = gid / x_inner; 
        const int off0 = halo_depth*(x + 1);
        const int index = off0 + col + row*x;

        double cell_vol = volume[index];
        double cell_mass = cell_vol*density[index];
        vol_shared[lid] = cell_vol;
        mass_shared[lid] = cell_mass;
        ie_shared[lid] = cell_mass*energy0[index];
        temp_shared[lid] = cell_mass*u[index];
    }

    __syncthreads();

#pragma unroll
    for(int ii = BLOCK_SIZE/2; ii > 0; ii /= 2)
    {
        if(lid < ii)
        {
            vol_shared[lid] += vol_shared[lid+ii];
            mass_shared[lid] += mass_shared[lid+ii];
            ie_shared[lid] += ie_shared[lid+ii];
            temp_shared[lid] += temp_shared[lid+ii];
        }

        __syncthreads();
    }

    vol_out[blockIdx.x] = vol_shared[0];
    mass_out[blockIdx.x] = mass_shared[0];
    ie_out[blockIdx.x] = ie_shared[0];
    temp_out[blockIdx.x] = temp_shared[0];
}
