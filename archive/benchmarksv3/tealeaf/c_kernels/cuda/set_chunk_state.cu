#include "../../settings.h"

__global__ void set_chunk_initial_state(
        const int x,
        const int y,
		const double default_energy, 
		const double default_density, 
		double* energy0,
	   	double* density)
{
	const int gid = threadIdx.x+blockDim.x*blockIdx.x;
	if(gid >= x*y) return;

    energy0[gid]=default_energy;
    density[gid]=default_density;
}	

__global__ void set_chunk_state(
        const int x,
        const int y,
        const double* vertex_x,
        const double* vertex_y,
        const double* cell_x,
        const double* cell_y,
        double* density,
        double* energy0,
        double* u,
        State state)
{
    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    const int x_loc = gid % x;
    const int y_loc = gid / x;
    int apply_state = 0;

    if(gid < x*y)
    {
        if(state.geometry == RECTANGULAR)
        {
            apply_state = (
                    vertex_x[x_loc+1] >= state.x_min && 
                    vertex_x[x_loc] < state.x_max    &&
                    vertex_y[y_loc+1] >= state.y_min &&
                    vertex_y[y_loc] < state.y_max);
        }
        else if(state.geometry == CIRCULAR)
        {
            double radius = sqrt(
                    (cell_x[x_loc]-state.x_min)*
                    (cell_x[x_loc]-state.x_min)+
                    (cell_y[y_loc]-state.y_min)*
                    (cell_y[y_loc]-state.y_min));

            apply_state = (radius <= state.radius);
        }
        else if(state.geometry == POINT)
        {
            apply_state = (
                    vertex_x[x_loc] == state.x_min &&
                    vertex_y[y_loc] == state.y_min);
        }

        // Check if state applies at this vertex, and apply
        if(apply_state)
        {
            energy0[gid] = state.energy;
            density[gid] = state.density;
        }
    }

    if(x_loc > 0 && x_loc < x-1 && 
            y_loc > 0 && y_loc < y-1)
    {
        u[gid]=energy0[gid]*density[gid];
    }
}
