#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdint.h>

#ifndef TSIZE
#define TSIZE 32
#endif

#define BSIZE 1024lu
//#define BSIZE 64

#define PSIZE 4096

constexpr size_t FPSIZE = (PSIZE/sizeof(float));


static __device__ __inline__ uint64_t __nano(){
  uint64_t mclk;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
  return mclk ;
}


__global__ void foo2(float* a, float* b, float* c, float* d)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    float temp = 0;

    if (gid < TSIZE)
    {
        #pragma unroll
        for (int tid = 0; tid < FPSIZE * TSIZE * 3; tid += FPSIZE)
        {
            //tid = i * FPSIZE;
            //ts1 = __nano();
           temp  += a[tid] + b[tid] + c[tid];
        }
        d[0] = temp;


        //printf("%lu, %lu, %lu\n", tid, ts1, ts2);
    }

}

__global__ void foo(float* a, float* b, float* c, float* d)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < TSIZE)
    {
        tid = tid * FPSIZE;
        float a1 = a[tid];
        __syncthreads();
        float b1 = b[tid];
        __syncthreads();
        float c1 = c[tid];
        __syncthreads();
        d[tid] = a1 + b1 + c1;
        __syncthreads();
        
        tid = tid + (FPSIZE * TSIZE);
        a1 = a[tid];
        __syncthreads();
        b1 = b[tid];
        __syncthreads();
        c1 = c[tid];
        __syncthreads();
        d[tid] = a1 + b1 + c1;
        __syncthreads();

        tid = tid + (FPSIZE * TSIZE);
        a1 = a[tid];
        __syncthreads();
        b1 = b[tid];
        __syncthreads();
        c1 = c[tid];
        __syncthreads();
        d[tid] = a1 + b1 + c1;
        __syncthreads();



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
    
    //printf("configuration: %lu, %lu\n", TSIZE/BSIZE, BSIZE); 
    //printf("TSIZE, BSIZE: %lu, %lu\n", TSIZE, BSIZE); 
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
    
    foo2<<<1, 1>>>(a, b, c, d);
    //foo<<<TSIZE/BSIZE, BSIZE>>>(a, b, c, d);
    cudaDeviceSynchronize();
    return 0;
}
