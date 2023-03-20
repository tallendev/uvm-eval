#pragma once

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__device__ inline double SUM(double a, double b)
{
	return a + b;
}

template < typename T, int offset >
class reduce
{
	public:
		__device__ inline static void run(T* array, T* out, T(*func)(T, T))
		{
			// only need to sync if not working within a warp
			if (offset > 16)
			{
				__syncthreads();
			}

			// only continue if it's in the lower half
			if (threadIdx.x < offset)
			{
				array[threadIdx.x] = func(array[threadIdx.x], array[threadIdx.x + offset]);
				reduce< T, offset/2 >::run(array, out, func);
			}
		}
};

template < typename T >
class reduce < T, 0 >
{
	public:
		__device__ inline static void run(T* array, T* out, T(*func)(T, T))
		{
			out[blockIdx.x] = array[0];
		}
};

void check_errors(int line_num, const char* file);
const char* cuda_codes(int code);
