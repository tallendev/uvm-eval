#include "comms.h"
#include "application.h"
#include "chunk.h"
#include "shared.h"
#include "drivers/drivers.h"
#include <cuda_runtime.h>

extern double total_walltime;
extern size_t UM_TOT_MEM;
extern size_t UM_MAX_MEM;
extern std::map<void*, size_t> MEMMAP;
void cudaFreeF(void* ptr);
void settings_overload(Settings* settings, int argc, char** argv);

int main(int argc, char** argv)
{
  // Immediately initialise MPI
  initialise_comms(argc, argv);

  barrier();

  // Create the settings wrapper
  Settings* settings = (Settings*)malloc(sizeof(Settings));
  set_default_settings(settings);

  // Fill in rank information
  initialise_ranks(settings);

  barrier();

  // Perform initialisation steps
  Chunk* chunks;
  initialise_application(&chunks, settings);

  settings_overload(settings, argc, argv);

  // Perform the solve using default or overloaded diffuse
#ifndef DIFFUSE_OVERLOAD
  diffuse(chunks, settings);
#else
  diffuse_overload(chunks, settings);
#endif

  printf("alloced,%ld\n", UM_MAX_MEM);
  // total avg time per cell
  printf("perf,%lf\n", total_walltime / (settings->grid_x_cells * settings->grid_y_cells));
  // Print the kernel-level profiling results
  if(settings->rank == MASTER)
  {
    PRINT_PROFILING_RESULTS(settings->kernel_profile);
  }

  // Finalise the kernel
  kernel_finalise_driver(chunks, settings);

  // Finalise each individual chunk
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    finalise_chunk(&(chunks[cc]));
    cudaFreeF(&(chunks[cc]));
  }

  // Finalise the application
  free(settings);
  finalise_comms();

  return EXIT_SUCCESS;
}

void settings_overload(Settings* settings, int argc, char** argv)
{
  for(int aa = 1; aa < argc; ++aa)
  {
    // Overload the solver
    if(strmatch(argv[aa], "-solver") ||
        strmatch(argv[aa], "--solver") || 
        strmatch(argv[aa], "-s"))
    {
      if(aa+1 == argc) break;
      if(strmatch(argv[aa+1], "cg")) settings->solver = CG_SOLVER;
      if(strmatch(argv[aa+1], "cheby")) settings->solver = CHEBY_SOLVER;
      if(strmatch(argv[aa+1], "ppcg")) settings->solver = PPCG_SOLVER;
      if(strmatch(argv[aa+1], "jacobi")) settings->solver = JACOBI_SOLVER;
    }
    else if(strmatch(argv[aa], "-x"))
    {
      if(aa+1 == argc) break;
      settings->grid_x_cells = atoi(argv[aa]);
    }
    else if(strmatch(argv[aa], "-y"))
    {
      if(aa+1 == argc) break;
      settings->grid_y_cells = atoi(argv[aa]);
    }
    else if(strmatch(argv[aa], "-help") ||
        strmatch(argv[aa], "--help") || 
        strmatch(argv[aa], "-h"))
    {
      print_and_log(settings, "tealeaf <options>\n");
      print_and_log(settings, "options:\n");
      print_and_log(settings, "\t-solver, --solver, -s:\n");
      print_and_log(settings, 
          "\t\tCan be 'cg', 'cheby', 'ppcg', or 'jacobi'\n");
      finalise_comms();
      exit(0);
    } 
  }
}
