#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdint.h>

#ifndef TSIZE
#define TSIZE 32
#endif

#ifndef BSIZE
#define BSIZE 32
#endif

#define PSIZE 4096

#define FPSIZE (4096/sizeof(float))


static __device__ __inline__ uint64_t __nano(){
  uint64_t mclk;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
  return mclk ;
}


__global__ void foo(float* a, float* b, float* c)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < TSIZE)
    {
        tid = tid * FPSIZE;
        a[tid] = 0;
        b[tid] = 0;
        c[tid] = 0;
        
        tid = tid + (FPSIZE * TSIZE);
        a[tid] = 0;
        b[tid] = 0;
        c[tid] = 0;
        
        tid = tid + (FPSIZE * TSIZE);
        a[tid] = 0;
        b[tid] = 0;
        c[tid] = 0;

        //printf("%lu, %lu, %lu\n", tid, ts1, ts2);
    }

}

size_t pad_2MB(size_t val)
{
    size_t ret = val / (PSIZE * 512);
    size_t diff = val % (PSIZE * 512);
    if (diff)
    {
        ret += 1;
    }
    return ret * (PSIZE * 512);
}

int main(void)
{
    float* a;
    float* b;
    float* c;

    size_t array_size = pad_2MB(3 * 2 * sizeof(float) * TSIZE * FPSIZE);
    assert(!cudaMallocManaged(&a, array_size));
    assert(!cudaMallocManaged(&b, array_size));
    assert(!cudaMallocManaged(&c, array_size));
    printf("array_size: %lu\n", array_size);
    
    for (size_t i = 0; i < TSIZE * FPSIZE; i++)
    {
        a[i] = i;
        b[i] = i;
        c[i] = i;
    }

    foo<<<TSIZE/BSIZE, BSIZE>>>(a, b, c);
    cudaDeviceSynchronize();
    return 0;
}
