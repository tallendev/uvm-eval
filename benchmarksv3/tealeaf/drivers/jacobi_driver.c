#include "../comms.h"
#include "drivers.h"
#include "../kernel_interface.h"
#include "../chunk.h"

// Performs a full solve with the Jacobi solver kernels
void jacobi_driver(
    Chunk* chunks, Settings* settings, 
    double rx, double ry, double* error)
{
  jacobi_init_driver(chunks, settings, rx, ry);

  // Iterate till convergence
  int tt;
  for(tt = 0; tt < settings->max_iters; ++tt)
  {
    jacobi_main_step_driver(chunks, settings, tt, error);

    halo_update_driver(chunks, settings, 1);

    if(fabs(*error) < settings->eps) break;
  }

  print_and_log(settings, "Jacobi: \t\t%d iterations\n", tt);
}

// Invokes the CG initialisation kernels
void jacobi_init_driver(
    Chunk* chunks, Settings* settings, double rx, double ry)
{
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_jacobi_init(&(chunks[cc]), settings, rx, ry);

      run_copy_u(&(chunks[cc]), settings);
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }

  // Need to update for the matvec
  reset_fields_to_exchange(settings);
  settings->fields_to_exchange[FIELD_U] = true;
}

// Invokes the main Jacobi solve kernels
void jacobi_main_step_driver(
    Chunk* chunks, Settings* settings, int tt, double* error)
{
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_jacobi_iterate(&(chunks[cc]), settings, error);
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }

  if(tt % 50 == 0)
  {
    halo_update_driver(chunks, settings, 1);

    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
      if(settings->kernel_language == C)
      {
        run_calculate_residual(&(chunks[cc]), settings);

        run_calculate_2norm(
            &(chunks[cc]), settings, chunks[cc].r, error);
      }
      else if(settings->kernel_language == FORTRAN)
      {
      }
    }
  }

  sum_over_ranks(settings, error);
}

