/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <cublas_v2.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

//#include "Timer.h"

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
        default: return "unknown error";
    }
}


__global__ void warmup()
{
    return;
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
                                                        float *B, size_t wA,
                                                        size_t wB) {
    // Block index
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

    // Thread index
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    size_t aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    size_t aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    size_t aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    size_t bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    size_t bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (size_t a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (size_t k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    size_t c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void sum_array(float* a, size_t a_size, const char* label)
{
    double sum = 0.0;
    //#pragma omp parallel  for reduction(+:sum) schedule (dynamic, 1) 
    for (size_t i = 0; i < a_size; ++i)
    {
        sum += a[i];
    }
    printf("%s: %lf\n", label, sum);
}

void ConstantInit(float *data, size_t size, float val) {
    //#pragma omp parallel for schedule (dynamic, 1)
    for (size_t i = 0; i < size; ++i) 
    {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
long MatrixMultiply(//int argc, char **argv,
                   size_t block_size, const dim3 &dimsA,
                   const dim3 &dimsB) {
    // Allocate host memory for matrices A and B
    size_t size_A = dimsA.x * dimsA.y;
    size_t mem_size_A = sizeof(float) * size_A;
    float *h_A;/* = reinterpret_cast<float *>(malloc(mem_size_A))*/;
    size_t size_B = dimsB.x * dimsB.y;
    size_t mem_size_B = sizeof(float) * size_B;
    float *h_B;/* = reinterpret_cast<float *>(malloc(mem_size_B))*/;

		checkCudaErrors(cudaMallocManaged(&h_A, mem_size_A));
		checkCudaErrors(cudaMallocManaged(&h_B, mem_size_B));

    // Initialize host memory
    const float valB = 0.01f;


    // Allocate device memory
    /*float *d_A, *d_B, *d_C;*/

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    size_t size_C = dimsC.x * dimsC.y;
    size_t mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C;/* = reinterpret_cast<float *>(malloc(mem_size_C));*/

		checkCudaErrors(cudaMallocManaged(&h_C, mem_size_C));

        printf("alloced,%lu\n", mem_size_A + mem_size_B + mem_size_C);
        printf("Allocated Mem: %lfGB\n", (mem_size_A + mem_size_B + mem_size_C) / (1024 * 1024 * 1024.));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);
    ConstantInit(h_C, size_C, 0);
/*
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
*/
		printf("Virtual memory addresses:\n\n") ;

    printf("Start address of matrix A: %p\n", &(h_A[0]));
    printf("End address of matrix A: %p\n\n", &(h_A[size_A-1]));

    printf("Start address of matrix B: %p\n", &(h_B[0]));
    printf("End address of matrix B: %p\n\n", &(h_B[size_B-1]));

    printf("Start address of matrix C: %p\n", &(h_C[0]));
    printf("End address of matrix C: %p\n\n", &(h_C[size_C-1]));

    printf("done\n");  
	
		// Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

     // Create a handle for CUBLAS


    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel 
    //MatrixMulCUDA<32> <<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    //MatrixMulCUDA<32> <<< grid, threads >>>(h_C, h_A, h_B, dimsA.x, dimsB.x);
		const float alpha = 1.;
    const float beta = 0.;
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;// = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimsA.x, dimsB.y, dimsA.y, &alpha, h_A, dimsA.x, h_B, dimsB.x, &beta, h_C, dimsC.x); 
/*
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS error: %s\n", cublasGetErrorString(status));
    }
    printf("done\n");
*/
  //  cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));

    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));
    size_t nIter = 1;

    cublasHandle_t handle[nIter];
    cudaStream_t pStream[nIter];
    for (size_t j = 0; j < nIter; j++)
    {
        cublasCreate(&handle[j]);
        cudaStreamCreate (&pStream[j]);
    }

    //Timer time;
    //time.start();
    // Execute the kernel
    
/*
    for (size_t j = 0; j < nIter; j++) {
        double sum = 0.0;
        #pragma omp parallel  
        {
            #pragma omp for reduction(+:sum) schedule (static, 512)
            for (size_t i = 0; i < size_A; ++i)
            {
                sum += h_A[i];
            }
            printf("%s: %lf\n", "h_A", sum);

            sum = 0.0;
            #pragma omp for reduction(+:sum) schedule (static, 512)
            for (size_t i = 0; i < size_B; ++i)
            {
                sum += h_B[i];
            }
            printf("%s: %lf\n", "h_B", sum);
            
            sum = 0.0;
            #pragma omp for reduction(+:sum) schedule (static, 512)
            for (size_t i = 0; i < size_C; ++i)
            {
                sum += h_C[i];
            }
            printf("%s: %lf\n", "h_C", sum);

            double val = 1;
            #pragma omp for schedule (static, 512)
            for (size_t i = 0; i < size_A; ++i) 
            {
                h_A[i] = val;
            }

            val = 0.1;
            #pragma omp for schedule (static, 512)
            for (size_t i = 0; i < size_B; ++i) 
            {
                h_B[i] = val;
            }
            
            val = 100;
            #pragma omp for schedule (static, 512)
            for (size_t i = 0; i < size_C; ++i) 
            {
                h_C[i] = val;
            }

            #pragma omp master
            {
                //cublasSetStream(handle[j], pStream[j]);
                status = cublasSgemm(handle[j], CUBLAS_OP_N, CUBLAS_OP_N, dimsA.x, dimsB.y, dimsA.y, &alpha, h_A, dimsA.x, h_B, dimsB.x, &beta, h_C, dimsC.x); 
                //status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimsA.x, dimsB.y, dimsA.y, &alpha, d_A, dimsA.x, d_B, dimsB.x, &beta, d_C, dimsC.x); 
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    printf("CUBLAS error: %s\n", cublasGetErrorString(status));
                }
                cudaDeviceSynchronize();
            }

        }
    }
*/
    //#pragma omp parallel for
    for (size_t j = 0; j < nIter; j++) {
        //cublasSetStream(handle[j], pStream[j]);
        status = cublasSgemm(handle[j], CUBLAS_OP_N, CUBLAS_OP_N, dimsA.x, dimsB.y, dimsA.y, &alpha, h_A, dimsA.x, h_B, dimsB.x, &beta, h_C, dimsC.x); 
        //status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimsA.x, dimsB.y, dimsA.y, &alpha, d_A, dimsA.x, d_B, dimsB.x, &beta, d_C, dimsC.x); 
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            printf("CUBLAS error: %s\n", cublasGetErrorString(status));
        }
        cudaDeviceSynchronize();
        /*
        sum_array(h_A, size_A, "A");
        sum_array(h_B, size_B, "B");
        sum_array(h_C, size_C, "C");
        ConstantInit(h_A, size_A, 1.0f);
        ConstantInit(h_B, size_B, valB);
        ConstantInit(h_C, size_C, 0);
        */
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    
    //time.stop();
    //std::cout << "time:" << time.elapsedSeconds() << std::endl;

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;

    printf("time,%f\n", msecPerMatrixMul/1000.0);
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                               static_cast<double>(dimsA.y) *
                               static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
                       (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops," \
        " WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);
    
    printf("perf,%f\n", gigaFlops);

    // Copy result from device to host
    //checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6;  // machine zero

/*
    for (size_t i = 0; i < static_cast<size_t>(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05lu]=%.8f, ref=%.8f error term is > %E\n",
                   i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }
*/

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
/*
		printf("Matrix A:");
            for(int i = 0; i < size_A; i++){
              if(i % dimsA.x == 0){
                printf("\n");
              }
              printf("%f ", h_A[i]);
            }
            printf("\n");

            printf("Matrix B:");
            for(int i = 0; i < size_B; i++){
              if(i % dimsB.x == 0){
                printf("\n");
              }
              printf("%f ", h_B[i]);
            }
            printf("\n");

            printf("Matrix C:");
            for(int i = 0; i < size_C; i++){
              if(i % dimsA.x == 0){
                printf("\n");
              }
              printf("%f ", h_C[i]);
            }
            printf("\n");

*/
    // Clean up memory
    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);
    /*checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));*/

    printf("\nNOTE: The CUDA Samples are not meant for performance"\
           "measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices" \
               " must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    int block_size = 32;

    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

    // width of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
                                               dimsB.x, dimsB.y);

    long matrix_result = MatrixMultiply(block_size, dimsA, dimsB);
    //long matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);

    exit(matrix_result);
}

