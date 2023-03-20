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

/*
__global__ void foo(float* a, float* b, float* c) {
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    tid = tid * FPSIZE;
    c[tid] = a[tid] + b[tid];
    tid = tid + (FPSIZE * TSIZE);
    c[tid] = a[tid] + b[tid];
    tid = tid + (FPSIZE * TSIZE);
    c[tid] = a[tid] + b[tid];
}
*/

__global__ void foo(float* a, float* b, float* c)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t ts1;
    size_t ts2;
    if (tid < TSIZE)
    {
        tid = tid * FPSIZE;
        //ts1 = __nano();
        c[tid] = a[tid] + b[tid];
        //ts2 = __nano();
        
        tid = tid + (FPSIZE * TSIZE);
        c[tid] = a[tid] + b[tid];
        
        tid = tid + (FPSIZE * TSIZE);
        c[tid] = a[tid] + b[tid];
        //printf("%lu, %lu, %lu\n", tid, ts1, ts2);
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
    float* b;
    float* c;

    assert(!cudaMallocManaged(&a, pad_2MB(2 * sizeof(float) * TSIZE * FPSIZE)));
    assert(!cudaMallocManaged(&b, pad_2MB(2 * sizeof(float) * TSIZE * FPSIZE)));
    assert(!cudaMallocManaged(&c, pad_2MB(2* sizeof(float) * TSIZE * FPSIZE)));
    
    //assert(!cudaMallocManaged(&a, sizeof(float) * TSIZE * FPSIZE));
    //assert(!cudaMallocManaged(&b, sizeof(float) * TSIZE * FPSIZE));
    //assert(!cudaMallocManaged(&c, sizeof(float) * TSIZE * FPSIZE));

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
