#include <stdlib.h>
#include "../../shared.h"
#include "mkl.h"

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
        double* ky,
        MKL_INT* a_row_index,
        MKL_INT* a_col_index,
        double* a_non_zeros)
{
    if(coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        die(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
    }

#pragma omp parallel for
    for(int jj = 0; jj < y; ++jj)
    {
        for(int kk = 0; kk < x; ++kk)
        {
            const int index = jj*x+kk;
            p[index] = 0.0;
            r[index] = 0.0;
            u[index] = energy[index]*density[index];
        }
    }

#pragma omp parallel for
    for(int jj = 1; jj < y-1; ++jj)
    {
        for(int kk = 1; kk < x-1; ++kk)
        {
            const int index = jj*x+kk;
            w[index] = (coefficient == CONDUCTIVITY) 
                ? density[index] : 1.0/density[index];
        }
    }

#pragma omp parallel for
    for(int jj = halo_depth; jj < y-1; ++jj)
    {
        for(int kk = halo_depth; kk < x-1; ++kk)
        {
            const int index = jj*x + kk;
            kx[index] = rx*(w[index-1]+w[index]) /
                (2.0*w[index-1]*w[index]);
            ky[index] = ry*(w[index-x]+w[index]) /
                (2.0*w[index-x]*w[index]);
        }
    }

    // Initialise the CSR sparse coefficient matrix
    for(MKL_INT jj = halo_depth; jj < y-1; ++jj)
    {
        for(MKL_INT kk = halo_depth; kk < x-1; ++kk)
        {
            const MKL_INT index = jj*x + kk;
            MKL_INT coef_index = a_row_index[index];

            if(jj >= halo_depth)
            {
                a_non_zeros[coef_index] = -ky[index];
                a_col_index[coef_index++] = index-x;
            }

            if(kk >= halo_depth)
            {
                a_non_zeros[coef_index] = -kx[index];
                a_col_index[coef_index++] = index-1;
            }

            a_non_zeros[coef_index] = (1.0 + 
                    kx[index+1] + kx[index] + 
                    ky[index+x] + ky[index]);
            a_col_index[coef_index++] = index;

            if(jj < y-halo_depth)
            {
                a_non_zeros[coef_index] = -ky[index+x];
                a_col_index[coef_index++] = index+x;
            }

            if(kk < x-halo_depth)
            {
                a_non_zeros[coef_index] = -kx[index+1];
                a_col_index[coef_index] = index+1;
            }
        }
    }

    double rro_temp = 0.0;

    MKL_INT m = x*y;
    mkl_cspblas_dcsrgemv(
            "n", &m, a_non_zeros, a_row_index, a_col_index, u, w);

    int x_inner = x-2*halo_depth;

#pragma omp parallel for reduction(+:rro_temp)
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        const int offset = jj*x + halo_depth;
        cblas_dcopy(x_inner, u + offset, 1, r + offset, 1);
        cblas_daxpy(x_inner, -1.0, w + offset, 1, r + offset, 1);
        cblas_dcopy(x_inner, r + offset, 1, p + offset, 1);
        rro_temp += cblas_ddot(x_inner, r + offset, 1, p + offset, 1);
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
        double* ky,
        MKL_INT* a_row_index,
        MKL_INT* a_col_index,
        double* a_non_zeros)
{
    double pw_temp = 0.0;

    MKL_INT m = x*y;
    mkl_cspblas_dcsrgemv(
            "n", &m, a_non_zeros, a_row_index, a_col_index, p, w);

    int x_inner = x - 2*halo_depth;
#pragma omp parallel for reduction(+:pw_temp)
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        int offset = jj*x + halo_depth;
        pw_temp += cblas_ddot(x_inner, w + offset, 1, p + offset, 1);
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
    int x_inner = x - 2*halo_depth;

#pragma omp parallel for reduction(+:rrn_temp)
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        const int offset = jj*x + halo_depth;
        cblas_daxpy(x_inner, alpha, p + offset, 1, u + offset, 1);
        cblas_daxpy(x_inner, -alpha, w + offset, 1, r + offset, 1);
        rrn_temp += cblas_ddot(x_inner, r + offset, 1, r + offset, 1);
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
    int x_inner = x - 2*halo_depth;

#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        const int offset = jj*x + halo_depth;
        cblas_dscal(x_inner, beta, p + offset, 1);
        cblas_daxpy(x_inner, 1.0, r + offset, 1, p + offset, 1);
    }
}

