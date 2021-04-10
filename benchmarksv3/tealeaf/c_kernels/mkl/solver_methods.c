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
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            int index = jj*x+kk;
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
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            int index = jj*x+kk;
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

#pragma omp parallel for reduction(+:norm_temp)
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            int index = jj*x+kk;
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
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            int index = jj*x+kk;
            energy[index] = u[index]/density[index];
        }
    }
}
