
__global__ void CuKnlSetField(
		double xCells,
		double yCells,
		double* energy0,
		double* energy1)
{
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
	energy1[gid] = energy0[gid];
}
