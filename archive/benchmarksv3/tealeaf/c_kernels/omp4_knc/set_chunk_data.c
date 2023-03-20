#include "../../settings.h"
#include "../../shared.h"

/*
 * 		SET CHUNK DATA KERNEL
 * 		Initialises the chunk's mesh data.
 */

// Extended kernel for the chunk initialisation
void set_chunk_data( 
        Settings* settings,
        int x,
        int y,
        int left,
        int bottom,
 	    double* cell_x,
		double* cell_y,
		double* vertex_x,
		double* vertex_y,
		double* volume,
		double* x_area,
		double* y_area)
{
    double x_min = settings->grid_x_min + settings->dx*(double)left;
    double y_min = settings->grid_y_min + settings->dy*(double)bottom;

	for(int ii = 0; ii < x+1; ++ii)
	{
		vertex_x[ii] = x_min + settings->dx*(ii-settings->halo_depth);
	}

	for(int ii = 0; ii < y+1; ++ii)
	{
		vertex_y[ii] = y_min + settings->dy*(ii-settings->halo_depth);
	}

	for(int ii = 0; ii < x; ++ii)
	{
		cell_x[ii] = 0.5*(vertex_x[ii]+vertex_x[ii+1]);
	}

	for(int ii = 0; ii < y; ++ii)
	{
		cell_y[ii] = 0.5*(vertex_y[ii]+vertex_y[ii+1]);
	}

	for(int ii = 0; ii < x*y; ++ii)
	{
		volume[ii] = settings->dx*settings->dy;
		x_area[ii] = settings->dy;
		y_area[ii] = settings->dx;
	}
}

