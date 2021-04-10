#include "../../kernel_interface.h"
#include "c_kernels.h"

// Initialisation kernels
void run_set_chunk_data(Chunk* chunk, Settings* settings)
{
    set_chunk_data(
            settings, chunk->x, chunk->y, chunk->left, chunk->bottom, 
            chunk->cell_x, chunk->cell_y, chunk->vertex_x, chunk->vertex_y,
            chunk->volume, chunk->x_area, chunk->y_area);
}

void run_set_chunk_state(Chunk* chunk, Settings* settings, State* states)
{
    set_chunk_state(chunk->x, chunk->y, chunk->vertex_x, chunk->vertex_y, 
            chunk->cell_x, chunk->cell_y, chunk->density, chunk->energy0, 
            chunk->u, settings->num_states, states);
}

void run_kernel_initialise(Chunk* chunk, Settings* settings)
{
    kernel_initialise(settings, chunk->x, chunk->y, &(chunk->density0), 
            &(chunk->density), &(chunk->energy0), &(chunk->energy), 
            &(chunk->u), &(chunk->u0), &(chunk->p), &(chunk->r), 
            &(chunk->mi), &(chunk->w), &(chunk->kx), &(chunk->ky), 
            &(chunk->sd), &(chunk->volume), 
            &(chunk->x_area), &(chunk->y_area), &(chunk->cell_x), 
            &(chunk->cell_y), &(chunk->cell_dx), &(chunk->cell_dy),
            &(chunk->vertex_dx), &(chunk->vertex_dy), &(chunk->vertex_x), 
            &(chunk->vertex_y), &(chunk->cg_alphas), &(chunk->cg_betas), 
            &(chunk->cheby_alphas), &(chunk->cheby_betas));
}

void run_kernel_finalise(
        Chunk* chunk, Settings* settings)
{
    kernel_finalise(
            chunk->density0, chunk->density, chunk->energy0, chunk->energy,
            chunk->u, chunk->u0, chunk->p, chunk->r, chunk->mi, chunk->w,
            chunk->kx, chunk->ky, chunk->sd, chunk->volume, chunk->x_area,
            chunk->y_area, chunk->cell_x, chunk->cell_y, chunk->cell_dx,
            chunk->cell_dy, chunk->vertex_dx, chunk->vertex_dy, chunk->vertex_x,
            chunk->vertex_y, chunk->cg_alphas, chunk->cg_betas,
            chunk->cheby_alphas, chunk->cheby_betas);
}

// Solver-wide kernels
void run_local_halos(
        Chunk* chunk, Settings* settings, int depth)
{
    START_PROFILING(settings->kernel_profile);
    local_halos(chunk->x, chunk->y, depth, settings->halo_depth, 
            chunk->neighbours, settings->fields_to_exchange, chunk->density,
            chunk->energy0, chunk->energy, chunk->u, chunk->p, 
            chunk->sd);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_pack_or_unpack(
        Chunk* chunk, Settings* settings, int depth,
        int face, bool pack, double* field, double* buffer)
{
    START_PROFILING(settings->kernel_profile);
    pack_or_unpack(chunk->x, chunk->y, depth, 
            settings->halo_depth, face, pack, field, buffer);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_store_energy(Chunk* chunk, Settings* settings)
{
    START_PROFILING(settings->kernel_profile);
    store_energy(chunk->x, chunk->y, chunk->energy0, chunk->energy);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_field_summary(
        Chunk* chunk, Settings* settings, 
        double* vol, double* mass, double* ie, double* temp)
{
    START_PROFILING(settings->kernel_profile);
    field_summary(chunk->x, chunk->y,
            settings->halo_depth, chunk->volume, chunk->density,
            chunk->energy0, chunk->u, vol, mass, ie, temp);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

// CG solver kernels
void run_cg_init(
        Chunk* chunk, Settings* settings, 
        double rx, double ry, double* rro)
{
    START_PROFILING(settings->kernel_profile);
    cg_init(chunk->x, chunk->y, 
            settings->halo_depth, settings->coefficient, rx, ry, 
            rro, chunk->density, chunk->energy, chunk->u, 
            chunk->p, chunk->r, chunk->w, 
            chunk->kx, chunk->ky);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_cg_calc_w(Chunk* chunk, Settings* settings, double* pw)
{
    START_PROFILING(settings->kernel_profile);
    cg_calc_w(chunk->x, chunk->y, 
            settings->halo_depth, pw, chunk->p, 
            chunk->w, chunk->kx,
            chunk->ky);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_cg_calc_ur(
        Chunk* chunk, Settings* settings, double alpha, double* rrn)
{
    START_PROFILING(settings->kernel_profile);
    cg_calc_ur(chunk->x, chunk->y, 
            settings->halo_depth, alpha, rrn, chunk->u, 
            chunk->p, chunk->r, chunk->w);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_cg_calc_p(Chunk* chunk, Settings* settings, double beta)
{
    START_PROFILING(settings->kernel_profile);
    cg_calc_p(chunk->x, chunk->y, 
            settings->halo_depth, beta, chunk->p, 
            chunk->r);
    STOP_PROFILING(settings->kernel_profile, __func__);
}


// Chebyshev solver kernels
void run_cheby_init(Chunk* chunk, Settings* settings)
{
    START_PROFILING(settings->kernel_profile);
    cheby_init(
            chunk->x, chunk->y, settings->halo_depth, 
            chunk->theta, chunk->u, chunk->u0, 
            chunk->p, chunk->r, chunk->w, 
            chunk->kx, chunk->ky);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_cheby_iterate(
        Chunk* chunk, Settings* settings, double alpha, double beta)
{
    START_PROFILING(settings->kernel_profile);
    cheby_iterate(
            chunk->x, chunk->y, settings->halo_depth, alpha, beta, 
            chunk->u, chunk->u0, chunk->p, chunk->r, chunk->w, 
            chunk->kx, chunk->ky); 
    STOP_PROFILING(settings->kernel_profile, __func__);
}


// Jacobi solver kernels
void run_jacobi_init(
        Chunk* chunk, Settings* settings, double rx, double ry)
{
    START_PROFILING(settings->kernel_profile);
    jacobi_init(chunk->x, chunk->y, settings->halo_depth, 
            settings->coefficient, rx, ry, chunk->density, chunk->energy, 
            chunk->u0, chunk->u, chunk->kx, chunk->ky);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_jacobi_iterate(
        Chunk* chunk, Settings* settings, double* error)
{
    START_PROFILING(settings->kernel_profile);
    jacobi_iterate(
            chunk->x, chunk->y, settings->halo_depth, error, chunk->kx, 
            chunk->ky, chunk->u0, chunk->u, chunk->r);
    STOP_PROFILING(settings->kernel_profile, __func__);
}


// PPCG solver kernels
void run_ppcg_init(Chunk* chunk, Settings* settings)
{
    START_PROFILING(settings->kernel_profile);
    ppcg_init(chunk->x, chunk->y, settings->halo_depth, chunk->theta, 
            chunk->r, chunk->sd);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_ppcg_inner_iteration(
        Chunk* chunk, Settings* settings, double alpha, double beta)
{
    START_PROFILING(settings->kernel_profile);
    ppcg_inner_iteration(
            chunk->x, chunk->y, settings->halo_depth, alpha, beta, chunk->u, 
            chunk->r, chunk->kx, chunk->ky, chunk->sd);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

// Shared solver kernels
void run_copy_u(Chunk* chunk, Settings* settings)
{
    START_PROFILING(settings->kernel_profile);
    copy_u(
            chunk->x, chunk->y, settings->halo_depth, chunk->u0, chunk->u);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_calculate_residual(Chunk* chunk, Settings* settings)
{
    START_PROFILING(settings->kernel_profile);
    calculate_residual(chunk->x, chunk->y, settings->halo_depth, chunk->u, 
            chunk->u0, chunk->r, chunk->kx, chunk->ky);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_calculate_2norm(
        Chunk* chunk, Settings* settings, double* buffer, double* norm)
{
    START_PROFILING(settings->kernel_profile);
    calculate_2norm(
            chunk->x, chunk->y, settings->halo_depth, buffer, norm);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_finalise(Chunk* chunk, Settings* settings)
{
    START_PROFILING(settings->kernel_profile);
    finalise(
            chunk->x, chunk->y, settings->halo_depth, chunk->energy, 
            chunk->density, chunk->u);
    STOP_PROFILING(settings->kernel_profile, __func__);
}

