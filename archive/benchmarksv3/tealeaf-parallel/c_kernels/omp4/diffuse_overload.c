#include "../../drivers/drivers.h"
#include "../../application.h"
#include "../../comms.h"

void solve(Chunk* chunks, Settings* settings, int tt, double* wallclock_prev);

// An implementation specific overload of the main timestep loop
void diffuse_overload(Chunk* chunks, Settings* settings)
{
    int n = chunks->x*chunks->y;

    print_and_log(settings,
            "This implementation overloads the diffuse function.\n");

    // Currently have to place all structure enclose pointers 
    // into local variables for OMP 4.0 to accept them in mapping clauses
    double* r = chunks->r;
    double* sd = chunks->sd;
    double* kx = chunks->kx;
    double* ky = chunks->ky;
    double* w = chunks->w;
    double* p = chunks->p;
    double* cheby_alphas = chunks->cheby_alphas;
    double* cheby_betas = chunks->cheby_betas;
    double* cg_alphas = chunks->cg_alphas;
    double* cg_betas = chunks->cg_betas;
    double* energy = chunks->energy;
    double* density = chunks->density;
    double* energy0 = chunks->energy0;
    double* density0 = chunks->density0;
    double* u = chunks->u;
    double* u0 = chunks->u0;

    settings->is_offload = true;

#pragma omp target data \
    map(to: r[:n], sd[:n], kx[:n], ky[:n], w[:n], \
            p[:n], cheby_alphas[:settings->max_iters], \
            cheby_betas[:settings->max_iters], cg_alphas[:settings->max_iters], \
            cg_betas[:settings->max_iters]) \
    map(tofrom: density[:n], energy[:n], density0[:n], energy0[:n], \
            u[:n], u0[:n])
    {
        double wallclock_prev = 0.0;
        for(int tt = 0; tt < settings->end_step; ++tt)
        {
            solve(chunks, settings, tt, &wallclock_prev);
        }
    }

    settings->is_offload = false;

    field_summary_driver(chunks, settings, true);
}

