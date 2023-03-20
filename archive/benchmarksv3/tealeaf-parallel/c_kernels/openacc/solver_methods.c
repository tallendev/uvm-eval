#include <stdlib.h>
#include "../../shared.h"

/*
 *		SHARED SOLVER METHODS
 */

// Copies the current u into u0
void copy_u(
        const int x,
        const int y,
        const int halo_depth,
        double* u0,
        double* u)
{
#pragma acc kernels loop independent \
    present(u0[:x*y], u[:x*y])
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            u0[index] = u[index];	
        }
    }
}

// Calculates the current value of r
void calculate_residual(
        const int x,
        const int y,
        const int halo_depth,
        double* u,
        double* u0,
        double* r,
        double* kx,
        double* ky)
{
#pragma acc kernels loop independent \
    present(kx[:x*y], ky[:x*y], u[:x*y], u0[:x*y], r[:x*y])
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            const double smvp = SMVP(u);
            r[index] = u0[index] - smvp;
        }
    }
}

// Calculates the 2 norm of a given buffer
void calculate_2norm(
        const int x,
        const int y,
        const int halo_depth,
        double* buffer,
        double* norm)
{
    double norm_temp = 0.0;

#pragma acc kernels loop independent \
    present(buffer[:x*y])
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            norm_temp += buffer[index]*buffer[index];			
        }
    }

    *norm += norm_temp;
}

// Finalises the solution
void finalise(
        const int x,
        const int y,
        const int halo_depth,
        double* energy,
        double* density,
        double* u)
{
#pragma acc kernels loop independent \
    present(energy[:x*y], u[:x*y], density[:x*y])
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            energy[index] = u[index]/density[index];
        }
    }
}
