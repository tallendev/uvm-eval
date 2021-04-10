#include "../chunk.h"
#include "../kernel_interface.h"

// Invokes the kernel initialisation kernels
void kernel_initialise_driver(Chunk* chunks, Settings* settings)
{
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_kernel_initialise(&(chunks[cc]), settings);
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }
}

// Invokes the kernel finalisation drivers
void kernel_finalise_driver(Chunk* chunks, Settings* settings)
{
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_kernel_finalise(&(chunks[cc]), settings);
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }
}
