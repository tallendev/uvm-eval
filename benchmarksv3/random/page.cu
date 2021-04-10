#include <stdio.h>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "safecuda.h"
#include <assert.h>
#include <random>
// num float in 4k page
#define PSIZE 1024lu //512lu //500 //1024 //512

#ifndef PNUM
#define PNUM  512 //524288 //6000000//6000000 //2097152
#endif
//#define PNUM 262144
#define ARRAY_SIZE (PSIZE * PNUM)

#ifndef THREADS
#define THREADS 64 //128
#endif



#ifndef TASKS_PER_THREAD
#define TASKS_PER_THREAD 100 //1
#endif

#ifndef BLOCKS
#define BLOCKS (1 + PNUM/THREADS) //i(PNUM/TASKS_PER_THREAD/THREADS)  //31250
#endif

//__device__ unsigned long long counter;


static inline void clflush2(volatile void *__p)
{
    asm volatile("clflush (%0)" :: "r" (__p));
}

__device__ unsigned long t1[PNUM];
//__device__ unsigned long t2[PNUM * PSIZE];
__device__ unsigned long tdif[PNUM];
__inline__ __device__ void prefetch_l1 (const void* addr)
{

      asm(" prefetch.global.L1 [ %1 ];": "=l"(addr) : "l"(addr));
}

__inline__ __device__ void prefetch_l2 (const void* addr)
{

      asm(" prefetch.global.L2 [ %1 ];": "=l"(addr) : "l"(addr));
}


static __device__ __inline__ uint64_t __nano(){
  uint64_t mclk;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
  return mclk ;
  }

static __device__ __inline__ uint32_t __myclock(){
  uint32_t mclk;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(mclk));
  return mclk ;}


extern "C"
__global__ void uvmer2(volatile float* a, const __restrict__ size_t* b)
{
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < PNUM)
    {
        float ele = a[b[idx] * PSIZE];
        if (ele == 0.35) 
        {
            //b[idx] = ele;
            a[idx] = b[idx + 73];
        }
    }
}

extern "C"
__global__ void uvmer(volatile float* a, float* b)
{
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < PNUM)
    {
        for (int i = 0; i < TASKS_PER_THREAD; i++)
        {
//            unsigned long timestamp = __nano(); //clock64(); //  __myclock();
            //double ele = a[idx * PSIZE];
//            printf("idx + i: %ld\n", idx + i);
            //double ele = a[(idx + i) * 100  * PSIZE];
            float ele = a[PSIZE * (idx + (i * 10) * (blockDim.x * gridDim.x))];
//            unsigned long timestamp2 = __nano();//clock64(); //__myclock();
            if (ele == 0.35) 
            {
                b[idx] = ele;
                //printf("%lf", ele);
            }
//            t1[idx] = timestamp;
            //t2[idx] = timestamp2;
//            tdif[idx] = timestamp2 - timestamp;
        }
    }
}

extern "C"
__global__ void stupid()
{
    return;
}

void printl(const char* const str)
{
    //printf("######## %s #######\n", str);
}


inline void copyIndexData(size_t* b)
{
    std::vector<size_t> indexes;
    indexes.reserve(PNUM);
    #pragma omp simd
    for (size_t i = 0; i < PNUM; i++)
    {
        //indexes[i] = i;
        indexes.push_back(i);
    }
    std::random_shuffle(indexes.begin(), indexes.end());
    cudaMemcpy(b, &indexes[0], PNUM * sizeof(size_t), cudaMemcpyHostToDevice);
}

int main(void)
{
    printl("init");
    float* array;
    float* array2;
 //   cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    typedef std::chrono::high_resolution_clock Clock;

    array2=NULL;

    CHECK_CUDA_ERROR();
    stupid<<<1,1>>>();
    cudaDeviceSynchronize();
    int khz;
    auto rng = std::default_random_engine {};

    for (int i = 0; i < 1; i++)
    {
        //std::shuffle(indexes.begin(), indexes.end(), rng);
        /*
        for (size_t i = 0; i < (PNUM > 100 ? 100 : PNUM); i++)
        {
            printf("i, indexes[i]: %lu, %lu\n", i, indexes[i]);
        }*/
        size_t* b;
        cudaMalloc(&b, PNUM * sizeof(size_t));
        copyIndexData(b);

        CHECK_CUDA_ERROR();
        printf("Allocating %u mb\n", ARRAY_SIZE * sizeof(float) / 100000);
        cudaMallocManaged(&array, ARRAY_SIZE * sizeof(float));
        CHECK_CUDA_ERROR();

        printf("alloced,%ld\n", ARRAY_SIZE * sizeof(float));
        #pragma simd
        for (size_t i = 0; i < ARRAY_SIZE; i++) 
        {
            array[i] = 0.0;
        }
        #pragma omp simd
        for (size_t i = 0; i < ARRAY_SIZE; i++) 
        {
            clflush2(array + i);
        }
        cudaEvent_t start;
        cudaEventCreate(&start);

        cudaEvent_t stop;
        cudaEventCreate(&stop);

        // Record the start event
        cudaEventRecord(start, NULL);

        uvmer2<<<BLOCKS, THREADS>>>(array, b);
        // Record the stop event
        cudaEventRecord(stop, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR();
        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);

        // should be pages / sec
        printf("perf,%lf\n", (BLOCKS * THREADS) / (msecTotal/1000.0));


        
    CHECK_CUDA_ERROR();
        cudaFree(array);
        cudaFree(b);
    CHECK_CUDA_ERROR();
    }
    CHECK_CUDA_ERROR();
}
