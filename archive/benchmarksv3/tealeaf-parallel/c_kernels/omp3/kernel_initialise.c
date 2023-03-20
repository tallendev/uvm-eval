#include "../../settings.h"
#include "../../shared.h"
#include <stdlib.h>

// Allocates, and zeroes an individual buffer
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
            const int index = kk + jj*x;
            (*a)[index] = 0.0;
        }
    }
}

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
        double** cheby_betas)
{
    print_and_log(settings,
            "Performing this solve with the OpenMP 3.0 %s solver\n",
            settings->solver_name);

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
}

void kernel_finalise(
        double* density0, double* density, double* energy0, double* energy, 
        double* u, double* u0, double* p, double* r, double* mi, 
        double* w, double* kx, double* ky, double* sd, 
        double* volume, double* x_area, double* y_area, double* cell_x, 
        double* cell_y, double* cell_dx, double* cell_dy, double* vertex_dx, 
        double* vertex_dy, double* vertex_x, double* vertex_y, 
        double* cg_alphas, double* cg_betas, double* cheby_alphas, 
        double* cheby_betas)
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
}
