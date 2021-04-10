#include "../../shared.h"

// Store original energy state
__global__ void store_energy(
        int x_inner,
        int y_inner,
        double* energy0,
        double* energy)
{
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    energy[gid] = energy0[gid];
}

