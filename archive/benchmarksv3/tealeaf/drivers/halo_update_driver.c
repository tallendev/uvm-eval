#include "../settings.h"
#include "../kernel_interface.h"
#include "drivers.h"

// Invoke the halo update kernels
void halo_update_driver(
    Chunk* chunks, Settings* settings, int depth)
{
  // Check that we actually have exchanges to perform
  if(!is_fields_to_exchange(settings)) return;

  remote_halo_driver(chunks, settings, depth);

  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_local_halos(&(chunks[cc]), settings, depth);
    }
    else if(settings->kernel_language == FORTRAN)
    {
      // Fortran store energy kernel
    }
  }
}
