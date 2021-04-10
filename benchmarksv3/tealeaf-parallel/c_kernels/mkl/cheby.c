#include "../../shared.h"
#include "mkl.h"

/*
 *		CHEBYSHEV SOLVER KERNEL
 */

// Calculates the new value for u.
void cheby_calc_u(
        const int x,
        const int y,
        const int halo_depth,
        double* u,
        double* p)
{
    int x_inner = x - 2*halo_depth;

#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        const int offset = jj*x + halo_depth;
        cblas_daxpy(x_inner, 1.0, p + offset, 1, u + offset, 1);
    }
}

// Initialises the Chebyshev solver
void cheby_init(
        const int x,
        const int y,
        const int halo_depth,
        const double theta,
        double* u,
        double* u0,
        double* p,
        double* r,
        double* w,
        double* kx,
        double* ky,
        MKL_INT* a_row_index,
        MKL_INT* a_col_index,
        double* a_non_zeros)
{
    MKL_INT m = x*y;

    mkl_cspblas_dcsrgemv(
            "n", &m, a_non_zeros, a_row_index, a_col_index, u, w);

    int x_inner = x - 2*halo_depth;

#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        const int offset = jj*x + halo_depth;
        cblas_dcopy(x_inner, u0 + offset, 1, r + offset, 1);
        cblas_daxpy(x_inner, -1.0, w + offset, 1, r + offset, 1);
        cblas_dscal(x_inner, 1.0/theta, r + offset, 1);
        cblas_dcopy(x_inner, r + offset, 1, p + offset, 1);
    }

    cheby_calc_u(x, y, halo_depth, u, p);
}

// The main chebyshev iteration
void cheby_iterate(
        const int x,
        const int y,
        const int halo_depth,
        double alpha,
        double beta,
        double* u,
        double* u0,
        double* p,
        double* r,
        double* w,
        double* kx,
        double* ky,
        MKL_INT* a_row_index,
        MKL_INT* a_col_index,
        double* a_non_zeros)
{
    MKL_INT m = x*y;

    mkl_cspblas_dcsrgemv(
            "n", &m, a_non_zeros, a_row_index, a_col_index, u, w);

    int x_inner = x - 2*halo_depth;

#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        const int offset = jj*x + halo_depth;
        cblas_dcopy(x_inner, u0 + offset, 1, r + offset, 1);
        cblas_daxpy(x_inner, -1.0, w + offset, 1, r + offset, 1);
        cblas_dscal(x_inner, alpha, p + offset, 1);
        cblas_daxpy(x_inner, beta, r + offset, 1, p + offset, 1);
    }

    cheby_calc_u(x, y, halo_depth, u, p);
}

