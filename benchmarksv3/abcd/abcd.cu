#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdint.h>

#ifndef TSIZE
#define TSIZE 32
#endif

#define BSIZE 32lu
//#define BSIZE 64

#define PSIZE 4096

constexpr size_t FPSIZE = (PSIZE/sizeof(float));


static __device__ __inline__ uint64_t __nano(){
  uint64_t mclk;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
  return mclk ;
}

__global__ void foo(float* a, float* b, float* c, float* d)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t ts1;
    size_t ts2;
    if (tid < TSIZE)
    {
        tid = tid * FPSIZE;
        //ts1 = __nano();
        d[tid] = a[tid] + b[tid] + c[tid];
        //ts2 = __nano();
        
        tid = tid + (FPSIZE * TSIZE);
        d[tid] = a[tid] + b[tid] + c[tid];
        
        tid = tid + (FPSIZE * TSIZE);
        d[tid] = a[tid] + b[tid] + c[tid];
        //printf("%lu, %lu, %lu\n", tid, ts1, ts2);
    }

}

#define SIZE_2MB (PSIZE * 512)
size_t pad_2MB(size_t val)
{
    size_t ret = (val / SIZE_2MB) * SIZE_2MB;
    size_t diff = val % SIZE_2MB;
    if (diff)
    {
        ret += SIZE_2MB;
    }
    return ret;
}

int main(void)
{
    float* a;
    float* b;
    float* c;
    float* d;
    
    printf("configuration: %lu, %lu\n", TSIZE/BSIZE, BSIZE); 
    printf("TSIZE, BSIZE: %lu, %lu\n", TSIZE, BSIZE); 
    printf("expected faults: %lu\n", 4 * 3 * TSIZE);

    printf("allocating %lu bytes per array\n", pad_2MB(2 * 3 * sizeof(float) * TSIZE * FPSIZE));
    assert(!cudaMallocManaged(&a, pad_2MB(2 * 3 * sizeof(float) * TSIZE * FPSIZE)));
    assert(!cudaMallocManaged(&b, pad_2MB(2 * 3 * sizeof(float) * TSIZE * FPSIZE)));
    assert(!cudaMallocManaged(&c, pad_2MB(2 * 3 * sizeof(float) * TSIZE * FPSIZE)));
    assert(!cudaMallocManaged(&d, pad_2MB(2 * 3 * sizeof(float) * TSIZE * FPSIZE)));
    
    //assert(!cudaMallocManaged(&a, sizeof(float) * TSIZE * FPSIZE));
    //assert(!cudaMallocManaged(&b, sizeof(float) * TSIZE * FPSIZE));
    //assert(!cudaMallocManaged(&c, sizeof(float) * TSIZE * FPSIZE));

    for (size_t i = 0; i < pad_2MB(2 * 3 * sizeof(float) * TSIZE * FPSIZE) / sizeof(float); i++)
    {
        a[i] = i;
        b[i] = i;
        c[i] = i;
        d[i] = i;
    }
    
    foo<<<TSIZE/BSIZE, BSIZE>>>(a, b, c, d);
    cudaDeviceSynchronize();
    return 0;
}
