#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdint.h>

#ifndef TSIZE
#define TSIZE 1
#endif

#define BSIZE 1

#define PSIZE 4096

#define FPSIZE (4096/sizeof(float))


static __device__ __inline__ uint64_t __nano(){
  uint64_t mclk;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
  return mclk ;
}

__global__ void foo(float* a)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < TSIZE)
    {
        printf("%f\n", a[tid]);
    }

}

size_t pad_2MB(size_t val)
{
    size_t ret = val / (PSIZE * 512);
    size_t diff = val % (PSIZE * 512);
    if (diff)
    {
        ret += (PSIZE * 512);
    }
    return ret;
}

int main(void)
{
    float* a;

    assert(!cudaMallocManaged(&a, pad_2MB(2 * sizeof(float) * TSIZE * FPSIZE)));
    
    //assert(!cudaMallocManaged(&a, sizeof(float) * TSIZE * FPSIZE));
    //assert(!cudaMallocManaged(&b, sizeof(float) * TSIZE * FPSIZE));
    //assert(!cudaMallocManaged(&c, sizeof(float) * TSIZE * FPSIZE));

    for (size_t i = 0; i < TSIZE * FPSIZE; i++)
    {
        a[i] = i + .5f;
    }

    foo<<<TSIZE/BSIZE, BSIZE>>>(a);
    cudaDeviceSynchronize();
    return 0;
}
