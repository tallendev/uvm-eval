#include "../../chunk.h"
#include "../../settings.h"

/*
 *      This is the main interface file for C based implementations.
 */

// Initialisation kernels
__global__ void set_chunk_data_vertices( 
        int x, int y, int halo_depth, double dx, double dy, double x_min,
        double y_min, double* vertex_x, double* vertex_y, double* vertex_dx,
		double* vertex_dy);

__global__ void set_chunk_data( 
        int x, int y, double dx, double dy, double* cell_x, double* cell_y,
 	    double* cell_dx, double* cell_dy, double* vertex_x, double* vertex_y,
		double* volume, double* x_area, double* y_area);

__global__ void set_chunk_initial_state(
        const int x, const int y, const double default_energy, 
        const double default_density, double* energy0, double* density);

__global__ void set_chunk_state(
        const int x, const int y, const double* vertex_x, const double* vertex_y,
        const double* cell_x, const double* cell_y, double* density, double* energy0,
        double* u, State state);

void kernel_initialise(
        Settings* settings, int x, int y, double** density0, 
        double** density, double** energy0, double** energy, double** u, 
        double** u0, double** p, double** r, double** mi, 
        double** w, double** kx, double** ky, double** sd, 
        double** volume, double** x_area, double** y_area, double** cell_x, 
        double** cell_y, double** cell_dx, double** cell_dy, double** vertex_dx, 
        double** vertex_dy, double** vertex_x, double** vertex_y,
        double** cg_alphas, double** cg_betas, double** cheby_alphas,
        double** cheby_betas, double** d_comm_buffer, double** d_reduce_buffer, 
        double** d_reduce_buffer2, double** d_reduce_buffer3, double** d_reduce_buffer4);

void kernel_finalise(
        double* cg_alphas, double* cg_betas, double* cheby_alphas,
        double* cheby_betas);

// Solver-wide kernels
void local_halos(
        const int x, const int y, const int halo_depth,
        const int depth, const int* chunk_neighbours,
        const bool* fields_to_exchange, double* density, double* energy0,
        double* energy, double* u, double* p, double* sd);

void pack_or_unpack(
        Chunk* chunk, Settings* settings, int depth, int face, 
        bool pack, double* field, double* buffer);

__global__ void store_energy(
        int x, int y, double* energy0, double* energy);

__global__ void field_summary(
		const int x_inner, const int y_inner, const int halo_depth,
		const double* volume, const double* density, const double* energy0,
		const double* u, double* vol_out, double* mass_out,
		double* ie_out, double* temp_out);

// CG solver kernels
__global__ void cg_init_u(
        const int x, const int y, const int coefficient,
        const double* density, const double* energy1, double* u,
        double* p, double* r, double* w);

__global__ void cg_init_k(
        const int x_inner, const int y_inner, const int halo_depth,
        const double* w, double* kx, double* ky, double rx, double ry);

__global__ void cg_init_others(
        const int x_inner, const int y_inner, const int halo_depth,
        const double* u, const double* kx, const double* ky,
        double* p, double* r, double* w, double* mi, double* rro);

__global__ void cg_calc_w(
        const int x_inner, const int y_inner, const int halo_depth,
        const double* kx, const double* ky, const double* p,
        double* w, double* pw);

__global__ void cg_calc_ur(
        const int x_inner, const int y_inner, const int halo_depth,
        const double alpha, const double* p, const double* w,
        double* u, double* r, double* rrn);

__global__ void cg_calc_p(
        const int x_inner, const int y_inner, const int halo_depth,
        const double beta, const double* r, double* p);

// Chebyshev solver kernels
__global__ void cheby_init(
        const int x_inner, const int y_inner, const int halo_depth,
        const double* u, const double* u0, const double* kx,
        const double* ky, const double theta, double* p,
        double* r, double* w);

__global__ void cheby_calc_u(
        const int x_inner, const int y_inner, const int halo_depth,
        const double* p, double* u);

__global__ void cheby_calc_p(
        const int x_inner, const int y_inner, const int halo_depth, const double* u,
        const double* u0, const double* kx, const double* ky,
        const double alpha, const double beta, double* p, double* r,
        double* w);

// Jacobi solver kernels
__global__ void jacobi_iterate(
        const int x_inner, const int y_inner, const int halo_depth,
        const double* kx, const double* ky, const double* u0,
        const double* r, double* u, double* error);

__global__ void jacobi_init(
		const int x_inner, const int y_inner, const int halo_depth,
		const double* density, const double* energy, const double rx,
		const double ry, double* kx, double* ky, double* u0,
		double* u, const int coefficient);

__global__ void jacobi_copy_u(
		const int x_inner, const int y_inner, const double* src, double* dest);

// PPCG solver kernels
__global__ void ppcg_init(
        const int x_inner, const int y_inner, const int halo_depth,
        const double theta, const double* r, double* sd);

__global__ void ppcg_calc_ur(
        const int x_inner, const int y_inner, const int halo_depth,
        const double* kx, const double* ky, const double* sd,
        double* u, double* r);

__global__ void ppcg_calc_sd(
        const int x_inner, const int y_inner, const int halo_depth,
        const double alpha, const double beta, const double* r, double* sd);

// Shared solver kernels
__global__ void copy_u(
		const int x_inner, const int y_inner, const int halo_depth,
		const double* src, double* dest);

__global__ void calculate_residual(
		const int x_inner, const int y_inner, const int halo_depth,
		const double* u, const double* u0, const double* kx,
		const double* ky, double* r);

__global__ void calculate_2norm(
		const int x_inner, const int y_inner, const int halo_depth,
		const double* src, double* norm);

__global__ void finalise(
        const int x_inner, const int y_inner, const int halo_depth,
        const double* density, const double* u, double* energy);

void sum_reduce_buffer(
        double* buffer, double* value, int len);

__global__ void zero_buffer(
        const int x, const int y, double* buffer);


