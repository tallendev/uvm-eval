
__global__ void set_chunk_data_vertices( 
        int x,
        int y,
        int halo_depth,
        double dx,
        double dy,
        double x_min,
        double y_min,
		double* vertex_x,
		double* vertex_y,
		double* vertex_dx,
		double* vertex_dy)
{
	const int gid = blockIdx.x*blockDim.x+threadIdx.x;

    if(gid < x+1)
	{
		vertex_x[gid] = x_min + dx*(gid-halo_depth);
		vertex_dx[gid] = dx;
	}

    if(gid < y+1)
	{
		vertex_y[gid] = y_min + dy*(gid-halo_depth);
		vertex_dy[gid] = dy;
	}
}

// Extended kernel for the chunk initialisation
__global__ void set_chunk_data( 
        int x,
        int y,
        double dx,
        double dy,
 	    double* cell_x,
		double* cell_y,
 	    double* cell_dx,
		double* cell_dy,
		double* vertex_x,
		double* vertex_y,
		double* volume,
		double* x_area,
		double* y_area)
{
	const int gid = blockIdx.x*blockDim.x+threadIdx.x;

    if(gid < x)
	{
		cell_x[gid] = 0.5*(vertex_x[gid]+vertex_x[gid+1]);
        cell_dx[gid] = dx;
	}

    if(gid < y)
	{
		cell_y[gid] = 0.5*(vertex_y[gid]+vertex_y[gid+1]);
        cell_dy[gid] = dy;
	}

    if(gid < x*y)
	{
		volume[gid] = dx*dy;
	}

    if(gid < (x+1)*y)
    {
        x_area[gid] = dy;
    }

    if(gid < x*(y+1))
    {
		y_area[gid] = dx;
    }
}

