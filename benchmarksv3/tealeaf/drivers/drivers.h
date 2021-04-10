#pragma once

#include "../chunk.h"

// Initialisation drivers
void set_chunk_data_driver(
        Chunk* chunk, Settings* settings);
void set_chunk_state_driver(
        Chunk* chunk, Settings* settings, State* states);
void kernel_initialise_driver(
        Chunk* chunks, Settings* settings);
void kernel_finalise_driver(
        Chunk* chunks, Settings* settings);

// Halo drivers
void halo_update_driver(
        Chunk* chunks, Settings* settings, int depth);
void remote_halo_driver(
        Chunk* chunks, Settings* settings, int depth);

// Conjugate Gradient solver drivers
void cg_driver(
        Chunk* chunks, Settings* settings, 
        double rx, double ry, double* error);
void cg_init_driver(
        Chunk* chunks, Settings* settings, 
        double rx, double ry, double* rro);
void cg_main_step_driver(
        Chunk* chunks, Settings* settings, int tt, 
        double* rro, double* error);

// Chebyshev solver drivers
void cheby_driver(
        Chunk* chunks, Settings* settings, 
        double rx, double ry, double* error);
void cheby_init_driver(
        Chunk* chunks, Settings* settings, int num_cg_iters, double* bb);
void cheby_coef_driver(
        Chunk* chunks, Settings* settings, int max_iters);
void cheby_main_step_driver(
        Chunk* chunks, Settings* settings, int cheby_iters, 
        bool is_calc_2norm, double* error);

// PPCG solver drivers
void ppcg_driver(
        Chunk* chunks, Settings* settings,
        double rx, double ry, double* error);
void ppcg_init_driver(
        Chunk* chunks, Settings* settings, double* rro);
void ppcg_main_step_driver(
        Chunk* chunks, Settings* settings, double* rro, double* error);

// Jacobi solver drivers
void jacobi_driver(
        Chunk* chunks, Settings* settings, 
        double rx, double ry, double* error);
void jacobi_init_driver(
        Chunk* chunks, Settings* settings, 
        double rx, double ry);
void jacobi_main_step_driver(
        Chunk* chunks, Settings* settings, int tt, double* error);

// Misc drivers
void field_summary_driver(
        Chunk* chunks, Settings* settings, bool solve_finished);
void store_energy_driver(
        Chunk* chunk, Settings* settings);
void solve_finished_driver(
        Chunk* chunks, Settings* settings);
void eigenvalue_driver_initialise(
        Chunk* chunks, Settings* settings, int num_cg_iters);

