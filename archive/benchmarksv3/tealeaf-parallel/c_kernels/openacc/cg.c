#include <stdlib.h>
#include "../../shared.h"

/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */

// Initialises the CG solver
void cg_init(
        const int x,
        const int y,
        const int halo_depth,
        const int coefficient,
        double rx,
        double ry,
        double* rro,
        double* density,
        double* energy,
        double* u,
        double* p,
        double* r,
        double* w,
        double* kx,
        double* ky)
{
    if(coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        die(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
    }

    int xy = x*y;

#pragma acc kernels loop independent collapse(2) \
    present(p[:xy], r[:xy], u[:xy], energy[:xy], density[:xy])
    for(int jj = 0; jj < y; ++jj)
    {
        for(int kk = 0; kk < x; ++kk)
        {
            const int index = kk + jj*x;
            p[index] = 0.0;
            r[index] = 0.0;
            u[index] = energy[index]*density[index];
        }
    }

#pragma acc kernels loop independent collapse(2) \
    present(w[:xy], density[:xy])
    for(int jj = 1; jj < y-1; ++jj)
    {
        for(int kk = 1; kk < x-1; ++kk)
        {
            const int index = kk + jj*x;
            w[index] = (coefficient == CONDUCTIVITY) 
                ? density[index] : 1.0/density[index];
        }
    }

#pragma acc kernels loop independent collapse(2) \
    present(kx[:xy], ky[:xy], w[:xy])
    for(int jj = halo_depth; jj < y-1; ++jj)
    {
        for(int kk = halo_depth; kk < x-1; ++kk)
        {
            const int index = kk + jj*x;
            kx[index] = rx*(w[index-1]+w[index]) /
                (2.0*w[index-1]*w[index]);
            ky[index] = ry*(w[index-x]+w[index]) /
                (2.0*w[index-x]*w[index]);
        }
    }

    double rro_temp = 0.0;

#pragma acc kernels loop independent collapse(2) \
    present(w[:xy], u[:xy], r[:xy], p[:xy], kx[:xy], ky[:xy])
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            const double smvp = SMVP(u);
            w[index] = smvp;
            r[index] = u[index]-w[index];
            p[index] = r[index];
            rro_temp += r[index]*p[index];
        }
    }

    // Sum locally
    *rro += rro_temp;
}

// Calculates w
void cg_calc_w(
        const int x,
        const int y,
        const int halo_depth,
        double* pw,
        double* p,
        double* w,
        double* kx,
        double* ky)
{
    double pw_temp = 0.0;

    int xy = x*y;

#pragma acc kernels loop independent collapse(2) \
    present(w[:xy], p[:xy], kx[:xy], ky[:xy])
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            const double smvp = SMVP(p);
            w[index] = smvp;
            pw_temp += w[index]*p[index];
        }
    }

    *pw += pw_temp;
}

// Calculates u and r
void cg_calc_ur(
        const int x,
        const int y,
        const int halo_depth,
        const double alpha,
        double* rrn,
        double* u,
        double* p,
        double* r,
        double* w)
{
    double rrn_temp = 0.0;

    int xy = x*y;

#pragma acc kernels loop independent \
    present(u[:xy], p[:xy], w[:xy], r[:xy])
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
#pragma acc loop independent
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;

            u[index] += alpha*p[index];
            r[index] -= alpha*w[index];
            rrn_temp += r[index]*r[index];
        }
    }

    *rrn += rrn_temp;
}

// Calculates p
void cg_calc_p(
        const int x,
        const int y,
        const int halo_depth,
        const double beta,
        double* p,
        double* r)
{
    int xy = x*y;

#pragma acc kernels loop independent \
    present(p[:xy], r[:xy])
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
#pragma acc loop independent
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;

            p[index] = beta*p[index] + r[index];
        }
    }
}

