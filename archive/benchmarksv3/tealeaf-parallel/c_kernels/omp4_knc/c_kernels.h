#include "../../settings.h"

/*
 *      This is the main interface file for C based implementations.
 */

// Initialisation kernels
void set_chunk_data(
        Settings* settings, int x, int y, int left,
        int bottom, double* cell_x, double* cell_y,
		double* vertex_x, double* vertex_y, double* volume,
		double* x_area, double* y_area);

void set_chunk_state(
        int x, int y, double* vertex_x, double* vertex_y, double* cell_x, 
        double* cell_y, double* density, double* energy0, double* u, 
        const int num_states, State* state);

void kernel_initialise(
        Settings* settings, int x, int y, double** density0, 
        double** density, double** energy0, double** energy, double** u, 
        double** u0, double** p, double** r, double** mi, 
        double** w, double** kx, double** ky, double** sd, 
        double** volume, double** x_area, double** y_area, double** cell_x, 
        double** cell_y, double** cell_dx, double** cell_dy, double** vertex_dx, 
        double** vertex_dy, double** vertex_x, double** vertex_y, 
        double** cg_alphas, double** cg_betas, double** cheby_alphas, 
        double** cheby_betas);

void kernel_finalise(
        double* density0, double* density, double* energy0, double* energy, 
        double* u, double* u0, double* p, double* r, double* mi, 
        double* w, double* kx, double* ky, double* sd, 
        double* volume, double* x_area, double* y_area, double* cell_x, 
        double* cell_y, double* cell_dx, double* cell_dy, double* vertex_dx, 
        double* vertex_dy, double* vertex_x, double* vertex_y, 
        double* cg_alphas, double* cg_betas, double* cheby_alphas, 
        double* cheby_betas);

// Solver-wide kernels
void local_halos(
        const int x, const int y, const int depth,
        const int halo_depth, const int* chunk_neighbours,
        const bool* fields_to_exchange, double* density, double* energy0,
        double* energy, double* u, double* p, double* sd,
        bool is_offload);

void pack_or_unpack(
        const int x, const int y, const int depth,
        const int halo_depth, const int face, bool pack, 
        double *field, double* buffer, bool is_offload);

void store_energy(
        int x, int y, double* energy0, double* energy);

void field_summary(
        const int x, const int y, const int halo_depth,
        double* volume, double* density, double* energy0, double* u,
        double* volOut, double* massOut, double* ieOut, double* tempOut);

// CG solver kernels
void cg_init(
        const int x, const int y, const int halo_depth,
        const int coefficient, double rx, double ry, double* rro,
        double* density, double* energy, double* u, double* p, 
        double* r, double* w, double* kx, double* ky);
void cg_calc_w(
        const int x, const int y, const int halo_depth, double* pw,
        double* p, double* w, double* kx, double* ky);

void cg_calc_ur(
        const int x, const int y, const int halo_depth,
        const double alpha, double* rrn, double* u, double* p,
        double* r, double* w);

void cg_calc_p(
        const int x, const int y, const int halo_depth,
        const double beta, double* p, double* r);

// Chebyshev solver kernels
void cheby_init(const int x, const int y,
        const int halo_depth, const double theta, double* u, double* u0,
        double* p, double* r, double* w, double* kx,
        double* ky);
void cheby_iterate(const int x, const int y,
        const int halo_depth, double alpha, double beta, double* u,
        double* u0, double* p, double* r, double* w,
        double* kx, double* ky); 

// Jacobi solver kernels
void jacobi_init(const int x, const int y,
        const int halo_depth, const int coefficient, double rx, double ry,
        double* density, double* energy, double* u0, double* u,
        double* kx, double* ky);
void jacobi_iterate(const int x, const int y,
        const int halo_depth, double* error, double* kx, double* ky, 
        double* u0, double* u, double* r);

// PPCG solver kernels
void ppcg_init(const int x, const int y, const int halo_depth,
        double theta, double* r, double* sd);
void ppcg_inner_iteration(const int x, const int y,
        const int halo_depth, double alpha, double beta, double* u,
        double* r, double* kx, double* ky,
        double* sd);

// Shared solver kernels
void copy_u(
        const int x, const int y, const int halo_depth, 
        double* u0, double* u);

void calculate_residual(
        const int x, const int y, const int halo_depth,
        double* u, double* u0, double* r, double* kx, 
        double* ky);

void calculate_2norm(
        const int x, const int y, const int halo_depth,
        double* buffer, double* norm);

void finalise(
        const int x, const int y, const int halo_depth,
        double* energy, double* density, double* u);

