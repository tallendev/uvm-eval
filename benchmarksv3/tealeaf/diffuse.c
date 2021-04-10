#include "comms.h"
#include "drivers/drivers.h"
#include "application.h"

double calc_dt(Chunk* chunks);
void calc_min_timestep(Chunk* chunks, double* dt, int chunks_per_task);
void solve(Chunk* chunks, Settings* settings, int tt, double* wallclock_prev);

// The main timestep loop
void diffuse(Chunk* chunks, Settings* settings)
{
  double wallclock_prev = 0.0;
  for(int tt = 0; tt < settings->end_step; ++tt)
  {
    solve(chunks, settings, tt, &wallclock_prev);
  }

  //field_summary_driver(chunks, settings, true);
}

double total_walltime = 0.0;

// Performs a solve for a single timestep
void solve(Chunk* chunks, Settings* settings, int tt, double* wallclock_prev)
{
  print_and_log(settings, "\nTimestep %d\n", tt+1);
  profiler_start_timer(settings->wallclock_profile);

  // Calculate minimum timestep information
  double dt = settings->dt_init;
  calc_min_timestep(chunks, &dt, settings->num_chunks_per_rank);

  // Pick the smallest timestep across all ranks
  min_over_ranks(settings, &dt);

  double rx = dt / (settings->dx*settings->dx);
  double ry = dt / (settings->dy*settings->dy);

  // Prepare halo regions for solve
  reset_fields_to_exchange(settings);
  settings->fields_to_exchange[FIELD_ENERGY1] = true;
  settings->fields_to_exchange[FIELD_DENSITY] = true;
  halo_update_driver(chunks, settings, 2);

  double error = 1e+10;

  // Perform the solve with one of the integrated solvers
  switch(settings->solver)
  {
    case JACOBI_SOLVER:
      jacobi_driver(chunks, settings, rx, ry, &error);
      break;
    case CG_SOLVER:
      cg_driver(chunks, settings, rx, ry, &error);
      break;
    case CHEBY_SOLVER:
      cheby_driver(chunks, settings, rx, ry, &error);
      break;
    case PPCG_SOLVER:
      ppcg_driver(chunks, settings, rx, ry, &error);
      break;
  }

  // Perform solve finalisation tasks
  solve_finished_driver(chunks, settings);

  if(tt % settings->summary_frequency == 0)
  {
    field_summary_driver(chunks, settings, false);
  }

  profiler_end_timer(settings->wallclock_profile, "Wallclock");

  double wallclock = settings->wallclock_profile->profiler_entries[0].time;
  print_and_log(settings, "Wallclock: \t\t%.3lfs\n", wallclock);
  print_and_log(settings, "Avg. time per cell: \t%.6e\n", 
      (wallclock-*wallclock_prev) /
      (settings->grid_x_cells *
       settings->grid_y_cells));
  print_and_log(settings, "Error: \t\t\t%.6e\n", error);

  total_walltime += (wallclock-*wallclock_prev);
}

// Calculate minimum timestep
void calc_min_timestep(Chunk* chunks, double* dt, int chunks_per_task)
{
  for(int cc = 0; cc < chunks_per_task; ++cc)
  {
    double dtlp = calc_dt(&(chunks[cc]));

    if(dtlp < *dt)
    {
      *dt = dtlp;
    }
  }
}

// Calculates a value for dt
double calc_dt(Chunk* chunk)
{
  // Currently defaults to config provided value
  return chunk->dt_init;
}

