#include "../comms.h"
#include "../chunk.h"
#include "../kernel_interface.h"
#include "drivers.h"

// Calls all kernels that wrap up a solve regardless of solver
void solve_finished_driver(Chunk* chunks, Settings* settings)
{
    double exact_error = 0.0;

    if(settings->check_result)
    {
        for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
        {
            if(settings->kernel_language == C)
            {
                run_calculate_residual(&(chunks[cc]), settings);

                run_calculate_2norm(
                        &(chunks[cc]), settings, chunks[cc].r, &exact_error);
            }
            else if(settings->kernel_language == FORTRAN)
            {
            }
        }

        sum_over_ranks(settings, &exact_error);
    }

    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        if(settings->kernel_language == C)
        {
            run_finalise(&(chunks[cc]), settings);
        }
        else if(settings->kernel_language == FORTRAN)
        {
        }
    }

    settings->fields_to_exchange[FIELD_ENERGY1] = true;
    halo_update_driver(chunks, settings, 1);
}

