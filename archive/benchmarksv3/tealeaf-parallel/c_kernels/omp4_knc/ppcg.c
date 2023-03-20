#include "../../shared.h"

/*
 *		PPCG SOLVER KERNEL
 */

// Initialises the PPCG solver
void ppcg_init(
        const int x,
        const int y,
        const int halo_depth,
        double theta,
        double* r,
        double* sd)
{
#pragma omp target
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            sd[index] = r[index] / theta;
        }
    }
}

// The PPCG inner iteration
void ppcg_inner_iteration(
        const int x,
        const int y,
        const int halo_depth,
        double alpha,
        double beta,
        double* u,
        double* r,
        double* kx,
        double* ky,
        double* sd)
{
#pragma omp target
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            const double smvp = SMVP(sd);
            r[index] -= smvp;
            u[index] += sd[index];
        }
    }

#pragma omp target
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            sd[index] = alpha*sd[index] + beta*r[index];
        }
    }
}

