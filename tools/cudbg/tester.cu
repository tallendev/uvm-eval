#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>


#define DUM_SIZE 4096

__device__ float dum[DUM_SIZE];

__global__ void foo()
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < DUM_SIZE)
    {
        dum[gid] = 15;
    }

    return;
}

__global__ void print_foo()
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < DUM_SIZE)
    {
        printf("val[%lu]: %f\n", gid, dum[gid]);
    }

    return;
}

int main(void)
{
    printf("Hello world\n");
    foo<<<DUM_SIZE/256,256>>>();
    printf("Sync device\n");
    cudaDeviceSynchronize();
    //print_foo<<<DUM_SIZE/256,256>>>();
    //cudaDeviceSynchronize();
    
    printf("Goodbye world\n");
    return 0;
}
