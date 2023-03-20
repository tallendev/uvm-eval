#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#define NVRTC_SAFE_CALL(x) \
        do { \
            nvrtcResult result = x; \
            if (result != NVRTC_SUCCESS) { \
                std::cerr << "\nerror: " #x " failed with error " \
                << nvrtcGetErrorString(result) << '\n'; \
            } \
         } while(0)

#define CUDA_SAFE_CALL(x) \
        do { \
            CUresult result = x; \
            if (result != CUDA_SUCCESS) { \
                const char *msg; \
                cuGetErrorName(result, &msg); \
                std::cerr << "\nerror: " #x " failed with error " \
                << msg << '\n'; \
                exit(1);\
            } \
        } while(0) 


#define CHECK_CUDA_ERROR()                                                    \
{                                                                             \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                   \
    {                                                                         \
        printf("error=%d name=%s at "                                         \
               "ln: %d\n  ",err,cudaGetErrorString(err),__LINE__);            \
        exit(1);\
    }                                                                         \
}

#define get_props(props)\
{\
    int count;\
    cudaGetDeviceCount(&count);\
    CHECK_CUDA_ERROR();\
    if (!count)\
    {\
        fprintf(stderr, "No devices found.\n");\
    }\
    cudaGetDeviceProperties(props, 0);\
    CHECK_CUDA_ERROR();\
}
