#include "mkl.h"
#include "../../settings.h"
#include "../../shared.h"
#include <stdlib.h>

void allocate_buffer(double** a, int x, int y);
void prepare_5pt_stencil_csr_rows(
        int x, int y, int halo_depth, MKL_INT* a_row_index);

// Allocates all of the field buffers
void kernel_initialise(
        Settings* settings, int x, int y, double** density0, 
        double** density, double** energy0, double** energy, double** u, 
        double** u0, double** p, double** r, double** mi, 
        double** w, double** kx, double** ky, double** sd, 
        double** volume, double** x_area, double** y_area, double** cell_x, 
        double** cell_y, double** cell_dx, double** cell_dy, double** vertex_dx, 
        double** vertex_dy, double** vertex_x, double** vertex_y, 
        double** cg_alphas, double** cg_betas, double** cheby_alphas,
        double** cheby_betas, MKL_INT** a_row_index, MKL_INT** a_col_index,
        double** a_non_zeros)
{
    allocate_buffer(density0, x, y);
    allocate_buffer(density, x, y);
    allocate_buffer(energy0, x, y);
    allocate_buffer(energy, x, y);
    allocate_buffer(u, x, y);
    allocate_buffer(u0, x, y);
    allocate_buffer(p, x, y);
    allocate_buffer(r, x, y);
    allocate_buffer(mi, x, y);
    allocate_buffer(w, x, y);
    allocate_buffer(kx, x, y);
    allocate_buffer(ky, x, y);
    allocate_buffer(sd, x, y);
    allocate_buffer(volume, x, y);
    allocate_buffer(x_area, x+1, y);
    allocate_buffer(y_area, x, y+1);
    allocate_buffer(cell_x, x, 1);
    allocate_buffer(cell_y, 1, y);
    allocate_buffer(cell_dx, x, 1);
    allocate_buffer(cell_dy, 1, y);
    allocate_buffer(vertex_dx, x+1, 1);
    allocate_buffer(vertex_dy, 1, y+1);
    allocate_buffer(vertex_x, x+1, 1);
    allocate_buffer(vertex_y, 1, y+1);
    allocate_buffer(cg_alphas, settings->max_iters, 1);
    allocate_buffer(cg_betas, settings->max_iters, 1);
    allocate_buffer(cheby_alphas, settings->max_iters, 1);
    allocate_buffer(cheby_betas, settings->max_iters, 1);

    // Zero-indexed three-array CSR variant
    *a_row_index = (MKL_INT*)malloc(sizeof(MKL_INT)*(x*y+1));

    prepare_5pt_stencil_csr_rows(x, y, settings->halo_depth, *a_row_index);

    MKL_INT num_non_zeros = (*a_row_index)[x*y];
    *a_col_index = (MKL_INT*)malloc(sizeof(MKL_INT)*num_non_zeros);
    *a_non_zeros = (double*)malloc(sizeof(double)*num_non_zeros);
}

// Allocates, and zeroes and individual buffer
void allocate_buffer(double** a, int x, int y)
{
    *a = (double*)malloc(sizeof(double)*x*y);

    if(*a == NULL) 
    {
        die(__LINE__, __FILE__, "Error allocating buffer %s\n");
    }

#pragma omp parallel for
    for(int jj = 0; jj < y; ++jj)
    {
        for(int kk = 0; kk < x; ++kk)
        {
            const int ind = jj*x+kk;
            (*a)[ind] = 0.0;
        }
    }
}

// Initialises the row index (CSR) for a 5pt stencil
void prepare_5pt_stencil_csr_rows(
        int x, int y, int halo_depth, MKL_INT* a_row_index)
{
    // Necessarily serialised row index calculation
    a_row_index[0] = 0;

    for(MKL_INT jj = 0; jj < y; ++jj)
    {
        for(MKL_INT kk = 0; kk < x; ++kk)
        {
            MKL_INT index = kk + jj*x;

            // Calculate position dependent row count
            MKL_INT row_count = 
                1 + (jj >= halo_depth) + (kk >= halo_depth) +  
                (jj < (y-halo_depth)) + (kk < (x-halo_depth));
            a_row_index[index+1] = a_row_index[index] + row_count;
        }
    }
}

void kernel_finalise(
        double* density0, double* density, double* energy0, double* energy, 
        double* u, double* u0, double* p, double* r, double* mi, 
        double* w, double* kx, double* ky, double* sd, 
        double* volume, double* x_area, double* y_area, double* cell_x, 
        double* cell_y, double* cell_dx, double* cell_dy, double* vertex_dx, 
        double* vertex_dy, double* vertex_x, double* vertex_y, 
        double* cg_alphas, double* cg_betas, double* cheby_alphas, 
        double* cheby_betas, MKL_INT* a_row_index, MKL_INT* a_col_index,
        double* a_non_zeros)
{
    free(density0);
    free(density);
    free(energy0);
    free(energy);
    free(u);
    free(u0);
    free(p);
    free(r);
    free(mi);
    free(w);
    free(kx);
    free(ky);
    free(sd);
    free(volume);
    free(x_area);
    free(y_area);
    free(cell_x);
    free(cell_y);
    free(cell_dx);
    free(cell_dy);
    free(vertex_dx);
    free(vertex_dy);
    free(vertex_x);
    free(vertex_y);
    free(cg_alphas);
    free(cg_betas);
    free(cheby_alphas);
    free(cheby_betas);
    free(a_row_index);
    free(a_col_index);
    free(a_non_zeros);
}
