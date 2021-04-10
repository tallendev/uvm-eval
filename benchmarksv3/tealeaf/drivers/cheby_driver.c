#include <float.h>
#include "../comms.h"
#include "drivers.h"
#include "../kernel_interface.h"
#include "../chunk.h"

void cheby_calc_est_iterations(
    Chunk* chunks, double error, double bb, int* est_iterations);

// Performs full solve with the Chebyshev kernels
void cheby_driver(
    Chunk* chunks, Settings* settings, 
    double rx, double ry, double* error)
{
  int tt;
  double rro = 0.0;
  int est_iterations = 0;
  int num_cheby_iters = 0;

  // Perform CG initialisation
  cg_init_driver(chunks, settings, rx, ry, &rro);

  // Iterate till convergence
  for(tt = 0; tt < settings->max_iters; ++tt)
  {
    // If we have already ran cheby iterations, continue
    // If we are error switching, check the error
    // If not error switching, perform preset iterations
    // Perform enough iterations to converge eigenvalues
    bool is_switch_to_cheby = (num_cheby_iters) || (settings->error_switch
        ? (*error < settings->eps_lim) && (tt > CG_ITERS_FOR_EIGENVALUES)
        : (tt > settings->presteps) && (*error < ERROR_SWITCH_MAX));

    if(!is_switch_to_cheby)
    {
      // Perform a CG iteration
      cg_main_step_driver(chunks, settings, tt, &rro, error);
    }
    else 
    {
      num_cheby_iters++;

      // Check if first step
      if(num_cheby_iters == 1)
      {
        // Initialise the solver
        double bb = 0.0;
        cheby_init_driver(chunks, settings, tt, &bb);

        // Perform the main step
        cheby_main_step_driver(
            chunks, settings, num_cheby_iters, true, error);

        // Estimate the number of Chebyshev iterations
        cheby_calc_est_iterations(chunks, *error, bb, &est_iterations);
      }
      else
      {
        bool is_calc_2norm = 
          (num_cheby_iters >= est_iterations) && ((tt+1) % 10 == 0);

        // Perform main step
        cheby_main_step_driver(
            chunks, settings, num_cheby_iters, is_calc_2norm, error);
      }
    }

    halo_update_driver(chunks, settings, 1);

    if(fabs(*error) < settings->eps) break;
  }

  print_and_log(settings, "CG: \t\t\t%d iterations\n", tt-num_cheby_iters+1);
  print_and_log(settings, 
      "Cheby: \t\t\t%d iterations (%d estimated)\n", 
      num_cheby_iters, est_iterations);
}

// Invokes the Chebyshev initialisation kernels
void cheby_init_driver(
    Chunk* chunks, Settings* settings, int num_cg_iters, double* bb)
{
  *bb = 0.0;

  // Initialise eigenvalues and Chebyshev coefficients
  eigenvalue_driver_initialise(chunks, settings, num_cg_iters);
  cheby_coef_driver(chunks, settings, settings->max_iters-num_cg_iters);

  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_calculate_2norm(&(chunks[cc]), settings, chunks[cc].u0, bb);

      run_cheby_init(&(chunks[cc]), settings);
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }

  reset_fields_to_exchange(settings);
  settings->fields_to_exchange[FIELD_U] = true;
  halo_update_driver(chunks, settings, 1);

  sum_over_ranks(settings, bb);
}

// Performs the main iteration step
void cheby_main_step_driver(
    Chunk* chunks, Settings* settings, int num_cheby_iters, 
    bool is_calc_2norm, double* error)
{
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_cheby_iterate(
          &(chunks[cc]), settings, 
          chunks[cc].cheby_alphas[num_cheby_iters], 
          chunks[cc].cheby_betas[num_cheby_iters]); 
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }

  if(is_calc_2norm)
  {
    *error = 0.0;

    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
      if(settings->kernel_language == C)
      {
        run_calculate_2norm(
            &(chunks[cc]), settings, chunks[cc].r, error);
      }
    }

    sum_over_ranks(settings, error);
  }
}

// Calculates the estimated iterations for Chebyshev solver
void cheby_calc_est_iterations(
    Chunk* chunks, double error, double bb, int* est_iterations)
{
  // Condition number is identical in all chunks/ranks
  double condition_number = chunks[0].eigmax / chunks[0].eigmin;

  // Calculate estimated iteration count
  double it_alpha = DBL_EPSILON*bb / (4.0*error);

  double gamm = 
    (sqrt(condition_number) - 1.0) / 
    (sqrt(condition_number) + 1.0);

  *est_iterations = roundf(logf(it_alpha) / (2.0*logf(gamm)));
}

// Calculates the Chebyshev coefficients for the chunk
void cheby_coef_driver(
    Chunk* chunks, Settings* settings, int max_iters)
{
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    chunks[cc].theta = (chunks[cc].eigmax + chunks[cc].eigmin) / 2.0;
    double delta = (chunks[cc].eigmax - chunks[cc].eigmin) / 2.0;
    double sigma = chunks[cc].theta / delta;
    double rho_old = 1.0 / sigma;

    for(int ii = 0; ii < max_iters; ++ii)
    {
      double rho_new = 1.0 / (2.0*sigma - rho_old);
      double cur_alpha = rho_new*rho_old;
      double cur_beta = 2.0*rho_new / delta;
      chunks[cc].cheby_alphas[ii] = cur_alpha;
      chunks[cc].cheby_betas[ii] = cur_beta;
      rho_old = rho_new;
    }
  }
}
