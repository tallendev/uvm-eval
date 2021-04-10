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
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid2;
    int tid3;
    size_t ts1;
    size_t ts2;
    size_t ts3;
    size_t ts4;
    size_t ts5;
    size_t ts6;
    size_t ts7;
    size_t ts8;
    size_t ts9;
    size_t ts10;
    float temp1;
    float temp2;
    if (tid < TSIZE)
    {
        tid = tid * FPSIZE;
        tid2 = tid + (FPSIZE * TSIZE);
        tid3 = tid2 + (FPSIZE * TSIZE);
        ts1 = __nano();
        temp1 = a[tid];
        ts2 = __nano();
        temp2 = b[tid];
        ts3 = __nano();
        c[tid] = temp1 + temp2;
        ts4 = __nano();

        temp1 = a[tid2];
        ts5 = __nano();
        temp2 = b[tid2];
        ts6 = __nano();
        c[tid2] = temp1 + temp2;
        ts7 = __nano();
        
        temp1 = a[tid3];
        ts8 = __nano();
        temp2 = b[tid3];
        ts9 = __nano();
        c[tid3] = temp1 + temp2;
        ts10 = __nano();
        printf("%u, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu\n", blockDim.x * blockIdx.x + threadIdx.x, ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9, ts10);
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
    size_t print_size;

    assert(!cudaMallocManaged(&a, pad_2MB(2 * sizeof(float) * TSIZE * FPSIZE)));
    assert(!cudaMallocManaged(&b, pad_2MB(2 * sizeof(float) * TSIZE * FPSIZE)));
    assert(!cudaMallocManaged(&c, pad_2MB(2* sizeof(float) * TSIZE * FPSIZE)));
    cudaDeviceGetLimit(&print_size, cudaLimitPrintfFifoSize);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, print_size * 100);
    
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
