#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <stdlib.h>
#include <stdio.h>

// Optional definitions
// CUDA_TALK: makes macros output error messages as provided by source (default = omitted for performance)

// Abort when error occurs (for CHECK_X() macros)
#ifdef ABORT_ON_CUDA_ERROR
#define SAFECUDA_ASSERT_ALWAYS_EXITS true
#else
#define SAFECUDA_ASSERT_ALWAYS_EXITS false
#endif

// Combination of techniques from:
// variable_args: https://stackoverflow.com/a/26408195
// zero_fix: https://gustedt.wordpress.com/2010/06/08/detect-empty-macro-arguments

// _last_one+2 will be too many arguments for the macros to handle
#define _ARG_N(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9,\
               _10, _11, _12, _13, _14, _15, _16, _17, _18, _19,\
               N, ...) N
// Universal functions (count, parens, pasting)
#define __COUNT_COMMA(...) _ARG_N(__VA_ARGS__)
#define _TRIGGER_PARENTHESIS_(...) ,
#define PASTE2(_0, _1) _PASTE2(_0, _1)
#define _PASTE2(_0, _1) _0 ## _1
#define PASTE5(_0, _1, _2, _3, _4) _PASTE5(_0, _1, _2, _3, _4)
#define _PASTE5(_0, _1, _2, _3, _4) _0 ## _1 ## _2 ## _3 ## _4
// Special comma cases for zeros
#define _COMMA_CASE_0001 0
#define _COMMA_CASE_0010 0
#define _COMMA_CASE_0100 0
#define _COMMA_CASE_1000 0
// Nonzero cases are a series of values-1 repeated #non-variadic-arguments times
#define _COMMA_CASE_0000 1
#define _COMMA_CASE_1111 2
#define _COMMA_CASE_2222 3
#define _COMMA_CASE_3333 4
// EACH __RESQ_N# MUST count backwards from maximum (same number of entries as _ARG_N)
  // All entries greater than max non-variadic arguments should be truncated
  // to the max non-variadic arugment value to simplify case-handling for variadic cases
// Max non-variadic arguments: 3
#define __RESQ_N3 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,\
                 3, 3, 3, 3, 3, 3, 3, 2, 1, 0
#define _COUNT_COMMA_N3(...) __COUNT_COMMA(__VA_ARGS__, __RESQ_N3)
#define VFUNC_NV3(func, ...) PASTE2(func, _VFUNC_NV3(__VA_ARGS__))(__VA_ARGS__)
#define _VFUNC_NV3(...) \
  PASTE5(_COMMA_CASE_,\
    _COUNT_COMMA_N3(__VA_ARGS__),\
    _COUNT_COMMA_N3(_TRIGGER_PARENTHESIS_ __VA_ARGS__),\
    _COUNT_COMMA_N3(__VA_ARGS__ ()),\
    _COUNT_COMMA_N3(_TRIGGER_PARENTHESIS_ __VA_ARGS__ ()))
// Max non-variadic arguments: 2
#define __RESQ_N2 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\
                 2, 2, 2, 2, 2, 2, 2, 2, 1, 0
#define _COUNT_COMMA_N2(...) __COUNT_COMMA(__VA_ARGS__, __RESQ_N2)
#define VFUNC_NV2(func, ...) PASTE2(func, _VFUNC_NV2(__VA_ARGS__))(__VA_ARGS__)
#define _VFUNC_NV2(...) \
  PASTE5(_COMMA_CASE_,\
    _COUNT_COMMA_N2(__VA_ARGS__),\
    _COUNT_COMMA_N2(_TRIGGER_PARENTHESIS_ __VA_ARGS__),\
    _COUNT_COMMA_N2(__VA_ARGS__ ()), _COUNT_COMMA_N2(_TRIGGER_PARENTHESIS_ __VA_ARGS__ ()))
// Max non-variadic arguments: 1
#define __RESQ_N1 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 0
#define _COUNT_COMMA_N1(...) __COUNT_COMMA(__VA_ARGS__, __RESQ_N1)
#define VFUNC_NV1(func, ...) PASTE2(func, _VFUNC_NV1(__VA_ARGS__))(__VA_ARGS__)
#define _VFUNC_NV1(...) \
  PASTE5(_COMMA_CASE_,\
    _COUNT_COMMA_N1(__VA_ARGS__),\
    _COUNT_COMMA_N1(_TRIGGER_PARENTHESIS_ __VA_ARGS__),\
    _COUNT_COMMA_N1(__VA_ARGS__ ()),\
    _COUNT_COMMA_N1(_TRIGGER_PARENTHESIS_ __VA_ARGS__ ()))
// EXEMPLAR DEFINITION FOR A SUPERMACRO:
// #define SUPERMACRO(...) VFUNC_NV#(SUPERMACRO, __VA_ARGS__) // Where # is the maximum number of non-variadic arguments
// Actual usage definitions go as follow (suppose _NV2)
// #define SUPERMACRO0() { what to do with 0 arguments }
// #define SUPERMACRO1(c1) { what to do with 1 argument }
// #define SUPERMACRO2(c1,c2) { what to do with 2 arguments }
// #define SUPERMACRO3(c1,c2,...) { what to do with first 2 arguments, __VA_ARGS__ for 3 and following }

// Better way to check CUDA API errors
#define CHECK_CUDA_ERROR(...) VFUNC_NV2(CHECK_CUDA_ERROR, __VA_ARGS__)
#define CUDA_ASSERT(...) VFUNC_NV2(CUDA_ASSERT, __VA_ARGS__)
// With quiet_macros, discard any/all buffer material from code/output
#ifndef CUDA_TALK
  #define CHECK_CUDA_ERROR3(cuda_code, buffer, ...) { cudaAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cuda_code); }
  #define CHECK_CUDA_ERROR2(cuda_code, constant) { cudaAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cuda_code); }
  #define CHECK_CUDA_ERROR1(cuda_code) { cudaAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cuda_code); }
  #define CHECK_CUDA_ERROR0() { cudaAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cudaGetLastError()); }
  #define CUDA_ASSERT3(cuda_code, buffer, ...) { cudaAssert(__FILE__, __LINE__, true, cuda_code); }
  #define CUDA_ASSERT2(cuda_code, constant) { cudaAssert(__FILE__, __LINE__, true, cuda_code); }
  #define CUDA_ASSERT1(cuda_code) { cudaAssert(__FILE__, __LINE__, true, cuda_code); }
  #define CUDA_ASSERT0() { cudaAssert(__FILE__, __LINE__, true, cudaGetLastError()); }
#else
  // Use additional information for debugging when errors occur
  // May supply (no, constant, variadic) amount of arguments)
  #define CHECK_CUDA_ERROR3(cuda_code, buffer, ...) { sprintf(buffer, __VA_ARGS__); cudaAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cuda_code, buffer); }
  #define CHECK_CUDA_ERROR2(cuda_code, constant) { cudaAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cuda_code, constant); }
  #define CHECK_CUDA_ERROR1(cuda_code) { cudaAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cuda_code); }
  #define CHECK_CUDA_ERROR0() { cudaAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cudaGetLastError()); }
  #define CUDA_ASSERT3(cuda_code, buffer, ...) { sprintf(buffer, __VA_ARGS__); cudaAssert(__FILE__, __LINE__, true, cuda_code, buffer); }
  #define CUDA_ASSERT2(cuda_code, constant) { cudaAssert(__FILE__, __LINE__, true, cuda_code, constant); }
  #define CUDA_ASSERT1(cuda_code) { cudaAssert(__FILE__, __LINE__, true, cuda_code); }
  #define CUDA_ASSERT0() { cudaAssert(__FILE__, __LINE__, true, cudaGetLastError()); }
#endif
inline void cudaAssert(const char* file, int line, bool abort,
                       cudaError_t code, const char* debug=NULL) {
  if (code != cudaSuccess) {
    if(debug == NULL) {
      fprintf(stderr, "CUDA Assertion Failed (%d) \"%s\" at %s:%d\n",
              code, cudaGetErrorString(code), file, line);
    }
    else {
      fprintf(stderr, "CUDA Assertion Failed (%d) \"%s\" at %s:%d\n%s\n",
              code, cudaGetErrorString(code), file, line, debug);
    }
    if(abort) exit(code);
  }
}

// Better way to check CUDA Kernel failures
#define CHECK_KERNEL_ERROR(...) VFUNC_NV1(CHECK_KERNEL_ERROR, __VA_ARGS__)
// With quiet_macros, discard any/all buffer material from code/output
#ifndef CUDA_TALK
  #define CHECK_KERNEL_ERROR2(buffer, ...) { kernelAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS); }
  #define CHECK_KERNEL_ERROR1(constant) { kernelAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS); }
  #define CHECK_KERNEL_ERROR0() { kernelAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS); }
#else
  // Use additional information for debugging when errors occur
  // May supply (no, constant, variadic) amount of arguments)
  #define CHECK_KERNEL_ERROR2(buffer, ...) { sprintf(buffer, __VA_ARGS__); kernelAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, buffer); }
  #define CHECK_KERNEL_ERROR1(constant) { kernelAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, constant); }
  #define CHECK_KERNEL_ERROR0() { kernelAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS); }
#endif
inline void kernelAssert(const char* file, int line, bool abort,
                         const char* debug=NULL) {
  cudaError_t code = cudaGetLastError();
  if(code != cudaSuccess) {
    if(debug == NULL) {
      fprintf(stderr, "CUDA Kernel Launch Failed (%d) \"%s\" at %s:%d\n",
              code, cudaGetErrorString(code), file, line);
    }
    else {
      fprintf(stderr, "CUDA Kernel Launch Failed (%d) \"%s\" at %s:%d\n%s\n",
              code, cudaGetErrorString(code), file, line, debug);
    }
    if(abort) exit(code);
  }
}

#define get_props(props, number) \
{ \
    int count; \
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&count)); \
    if (count < number) fprintf(stderr, "CUDA Device %d not found\n", number); \
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(props, number)); \
}

#ifndef CUDA_TALK
#define NVRTC_SAFE_CALL(x) x
#define CUDA_SAFE_CALL(x) \
        do { \
            CUresult result = x; \
            if (result != CUDA_SUCCESS) { \
               exit(1);\
            } \
        } while(0)
#else
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
               fprintf(stderr, "\nERROR: %d failed with error %s", x, msg); \
               exit(1);\
            } \
        } while(0)
#endif

/* SHOULD ONLY LOAD IF YOU #include <cublas_v2.h> IN YOUR CODE */
#ifdef CUBLAS_V2_H_
// There isn't an equivalent of cudaGetErrorString for cublas... -_-
const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "Unknown Error";
}
// Better way to check CUBLAS API errors
#define CHECK_CUBLAS_ERROR(...) VFUNC_NV2(CHECK_CUBLAS_ERROR, __VA_ARGS__)
#define CUBLAS_ASSERT(...) VFUNC_NV2(CUBLAS_ASSERT, __VA_ARGS__)
// With quiet_macros, discard any/all buffer material from code/output
#ifndef CUDA_TALK
  #define CHECK_CUBLAS_ERROR3(cublas_code, buffer, ...) { cublasAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cublas_code); }
  #define CHECK_CUBLAS_ERROR2(cublas_code, constant) { cublasAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cublas_code); }
  #define CHECK_CUBLAS_ERROR1(cublas_code) { cublasAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cublas_code); }
  #define CHECK_CUBLAS_ERROR0() { cublasAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cublasGetError()); }
  #define CUBLAS_ASSERT3(cublas_code, buffer, ...) { cublasAssert(__FILE__, __LINE__, true, cublas_code); }
  #define CUBLAS_ASSERT2(cublas_code, constant) { cublasAssert(__FILE__, __LINE__, true, cublas_code); }
  #define CUBLAS_ASSERT1(cublas_code) { cublasAssert(__FILE__, __LINE__, true, cublas_code); }
  #define CUBLAS_ASSERT0() { cublasAssert(__FILE__, __LINE__, true, cublasGetError()); }
#else
  // Use additional information for debugging when errors occur
  // May supply (no, constant, variadic) amount of arguments)
  #define CHECK_CUBLAS_ERROR3(cublas_code, buffer, ...) { sprintf(buffer, __VA_ARGS__); cublasAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cublas_code, buffer); }
  #define CHECK_CUBLAS_ERROR2(cublas_code, constant) { cublasAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cublas_code, constant); }
  #define CHECK_CUBLAS_ERROR1(cublas_code) { cublasAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cublas_code); }
  #define CHECK_CUBLAS_ERROR0() { cublasAssert(__FILE__, __LINE__, SAFECUDA_ASSERT_ALWAYS_EXITS, cublasGetError()); }
  #define CUBLAS_ASSERT3(cublas_code, buffer, ...) { sprintf(buffer, __VA_ARGS__); cublasAssert(__FILE__, __LINE__, true, cublas_code, buffer); }
  #define CUBLAS_ASSERT2(cublas_code, constant) { cublasAssert(__FILE__, __LINE__, true, cublas_code, constant); }
  #define CUBLAS_ASSERT1(cublas_code) { cublasAssert(__FILE__, __LINE__, true, cublas_code); }
  #define CUBLAS_ASSERT0() { cublasAssert(__FILE__, __LINE__, true, cublasGetError()); }
#endif
inline void cublasAssert(const char* file, int line, bool abort,
                         cublasStatus_t code, const char* debug=NULL) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    if(debug == NULL) {
      fprintf(stderr, "CUBLAS Assertion Failed (%d) \"%s\" at %s:%d\n",
              code, cublasGetErrorString(code), file, line);
    }
    else {
      fprintf(stderr, "CUBLAS Assertion Failed (%d) \"%s\" at %s:%d\n%s\n",
              code, cublasGetErrorString(code), file, line, debug);
    }
    if(abort) exit(code);
  }
}
#endif

/* SHOULD ONLY LOAD IF YOU #include <nvshmem.h> IN YOUR CODE**/
#ifdef _NVSHMEM_H_
// The api is different
#endif

