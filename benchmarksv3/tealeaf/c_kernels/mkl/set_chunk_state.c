#include <math.h>
#include "../../settings.h"

/*
 *      SET CHUNK STATE KERNEL
 *		Sets up the chunk geometry.
 */

// Entry point for set chunk state kernel
void set_chunk_state(
        int x,
        int y,
        double* vertex_x,
        double* vertex_y,
        double* cell_x,
        double* cell_y,
        double* density,
        double* energy0,
        double* u,
        const int num_states,
        State* states)
{
    // Set the initial state
    for(int ii = 0; ii != x*y; ++ii)
    {
        energy0[ii] = states[0].energy;
        density[ii] = states[0].density;
    }	

    // Apply all of the states in turn
    for(int ss = 1; ss < num_states; ++ss)
    {
        for(int jj = 0; jj != y; ++jj) 
        {
            for(int kk = 0; kk != x; ++kk) 
            {
                int apply_state = 0;

                if(states[ss].geometry == RECTANGULAR)
                {
                    apply_state = (
                            vertex_x[kk+1] >= states[ss].x_min && 
                            vertex_x[kk] < states[ss].x_max    &&
                            vertex_y[jj+1] >= states[ss].y_min &&
                            vertex_y[jj] < states[ss].y_max);
                }
                else if(states[ss].geometry == CIRCULAR)
                {
                    double radius = sqrt(
                            (cell_x[kk]-states[ss].x_min)*
                            (cell_x[kk]-states[ss].x_min)+
                            (cell_y[jj]-states[ss].y_min)*
                            (cell_y[jj]-states[ss].y_min));

                    apply_state = (radius <= states[ss].radius);
                }
                else if(states[ss].geometry == POINT)
                {
                    apply_state = (
                            vertex_x[kk] == states[ss].x_min &&
                            vertex_y[jj] == states[ss].y_min);
                }

                // Check if state applies at this vertex, and apply
                if(apply_state)
                {
                    int index = jj*x+kk;
                    energy0[index] = states[ss].energy;
                    density[index] = states[ss].density;
                }
            }
        }
    }

    // Set an initial state for u
    for(int jj = 1; jj != y-1; ++jj) 
    {
        for(int kk = 1; kk != x-1; ++kk) 
        {
            int index = jj*x+kk;
            u[index] = energy0[index]*density[index];
        }
    }
}

