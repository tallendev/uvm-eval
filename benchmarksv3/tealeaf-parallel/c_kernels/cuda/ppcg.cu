
__global__ void ppcg_init(
		const int x_inner,
		const int y_inner,
        const int halo_depth,
		const double theta,
		const double* r,
		double* sd)
{
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
	if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
	const int col = gid % x_inner;
	const int row = gid / x_inner; 
	const int off0 = halo_depth*(x + 1);
	const int index = off0 + col + row*x;

	sd[index] = r[index] / theta;
}

__global__ void ppcg_calc_ur(
		const int x_inner,
		const int y_inner,
        const int halo_depth,
		const double* kx,
		const double* ky,
		const double* sd,
		double* u,
		double* r)
{
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
	if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
	const int col = gid % x_inner;
	const int row = gid / x_inner; 
	const int off0 = halo_depth*(x + 1);
	const int index = off0 + col + row*x;

	const double smvp = (1.0
			+ (kx[index+1]+kx[index])
			+ (ky[index+x]+ky[index]))*sd[index]
		- (kx[index+1]*sd[index+1]+kx[index]*sd[index-1])
		- (ky[index+x]*sd[index+x]+ky[index]*sd[index-x]);

	r[index] -= smvp;
	u[index] += sd[index];
}

__global__ void ppcg_calc_sd(
		const int x_inner,
		const int y_inner,
        const int halo_depth,
		const double alpha,
		const double beta,
		const double* r,
		double* sd)
{
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
	if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
	const int col = gid % x_inner;
	const int row = gid / x_inner; 
	const int off0 = halo_depth*(x + 1);
	const int index = off0 + col + row*x;

	sd[index] = alpha*sd[index] + beta*r[index];
}

