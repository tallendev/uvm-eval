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


#define DEVICE_STATIC_INTRINSIC_QUALIFIERS  static __device__ __forceinline__

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define PXL_GLOBAL_PTR   "l"
#else
#define PXL_GLOBAL_PTR   "r"
#endif

DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_l1(const void* const ptr)
{
  asm("prefetch.global.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_uniform(const void* const ptr)
{
  asm("prefetchu.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_l2(const void* const ptr)
{
  asm("prefetch.global.L2 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

static __device__ __inline__ uint64_t __nano(){
  uint64_t mclk;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
  return mclk ;
}

__global__ void foo(float* a, float* b, float* c)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tid1 = tid * FPSIZE;
    size_t tid2 = tid1 + (FPSIZE * TSIZE);
    size_t tid3 = tid2 + (FPSIZE * TSIZE);
    if (tid < TSIZE)
    {
        __prefetch_global_l2(&a[tid1]);
        __prefetch_global_l2(&b[tid1]);
        __prefetch_global_l2(&c[tid1]);
        c[tid1] = a[tid1] + b[tid1];
        __prefetch_global_l2(&a[tid2]);
        __prefetch_global_l2(&b[tid2]);
        __prefetch_global_l2(&c[tid2]);
        c[tid2] = a[tid2] + b[tid2];
        __prefetch_global_l2(&a[tid3]);
        __prefetch_global_l2(&b[tid3]);
        __prefetch_global_l2(&c[tid3]);
        c[tid3] = a[tid3] + b[tid3];
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
