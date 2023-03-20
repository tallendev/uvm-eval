#include "../comms.h"
#include "drivers.h"
#include "../kernel_interface.h"
#include "../chunk.h"

void ppcg_inner_iterations(
    Chunk* chunks, Settings* settings);

// Performs a full solve with the PPCG solver
void ppcg_driver(Chunk* chunks, Settings* settings,
    double rx, double ry, double* error)
{
  int tt;
  double rro = 0.0;
  int num_ppcg_iters = 0;

  // Perform CG initialisation
  cg_init_driver(chunks, settings, rx, ry, &rro);

  // Iterate till convergence
  for(tt = 0; tt < settings->max_iters; ++tt)
  {
    // If we have already ran PPCG inner iterations, continue
    // If we are error switching, check the error
    // If not error switching, perform preset iterations
    // Perform enough iterations to converge eigenvalues
    bool is_switch_to_ppcg = (num_ppcg_iters) || (settings->error_switch
        ? (*error < settings->eps_lim) && (tt > CG_ITERS_FOR_EIGENVALUES)
        : (tt > settings->presteps) && (*error < ERROR_SWITCH_MAX));

    if(!is_switch_to_ppcg)
    {
      // Perform a CG iteration
      cg_main_step_driver(chunks, settings, tt, &rro, error);
    }
    else 
    {
      num_ppcg_iters++;

      // If first step perform initialisation
      if(num_ppcg_iters == 1)
      {
        // Initialise the eigenvalues and Chebyshev coefficients
        eigenvalue_driver_initialise(chunks, settings, tt);
        cheby_coef_driver(
            chunks, settings, settings->ppcg_inner_steps);

        ppcg_init_driver(chunks, settings, &rro);
      }

      ppcg_main_step_driver(chunks, settings, &rro, error);
    }

    halo_update_driver(chunks, settings, 1);

    if(fabs(*error) < settings->eps) break;
  }

  print_and_log(settings, "CG: \t\t\t%d iterations\n", tt-num_ppcg_iters+1);
  print_and_log(settings, 
      "PPCG: \t\t\t%d iterations (%d inner iterations per)\n", 
      num_ppcg_iters, settings->ppcg_inner_steps);
}

// Invokes the PPCG initialisation kernels
void ppcg_init_driver(Chunk* chunks, Settings* settings, double* rro)
{
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_calculate_residual(&(chunks[cc]), settings);
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }

  reset_fields_to_exchange(settings);
  settings->fields_to_exchange[FIELD_P] = true;
  halo_update_driver(chunks, settings, 1);

  sum_over_ranks(settings, rro);
}

// Invokes the main PPCG solver kernels
void ppcg_main_step_driver(
    Chunk* chunks, Settings* settings, double* rro, double* error)
{
  double pw = 0.0;

  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_cg_calc_w(&(chunks[cc]), settings, &pw);
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }

  sum_over_ranks(settings, &pw);

  double alpha = *rro / pw;
  double rrn = 0.0;

  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_cg_calc_ur(&(chunks[cc]), settings, alpha, &rrn);
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }

  // Perform the inner iterations
  ppcg_inner_iterations(chunks, settings);

  rrn = 0.0;
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_calculate_2norm(
          &(chunks[cc]), settings, chunks[cc].r, &rrn);
    }
  }

  sum_over_ranks(settings, &rrn);

  double beta = rrn / *rro;

  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_cg_calc_p(&(chunks[cc]), settings, beta);
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }

  *error = rrn;
  *rro = rrn;
}

// Performs the inner iterations of the PPCG solver
void ppcg_inner_iterations(
    Chunk* chunks, Settings* settings)
{
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_ppcg_init(&(chunks[cc]), settings);
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }

  reset_fields_to_exchange(settings);
  settings->fields_to_exchange[FIELD_SD] = true;

  for(int pp = 0; pp < settings->ppcg_inner_steps; ++pp)
  {
    halo_update_driver(chunks, settings, 1);

    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
      if(settings->kernel_language == C)
      {
        run_ppcg_inner_iteration(
            &(chunks[cc]), settings, chunks[cc].cheby_alphas[pp], 
            chunks[cc].cheby_betas[pp]);
      }
      else if(settings->kernel_language == FORTRAN)
      {
      }
    }
  }

  reset_fields_to_exchange(settings);
  settings->fields_to_exchange[FIELD_P] = true;
}
