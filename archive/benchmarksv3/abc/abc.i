# 1 "abc.cu"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 1
# 61 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
#pragma GCC diagnostic push


#pragma GCC diagnostic ignored "-Wunused-function"
# 83 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_config.h" 1
# 208 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_config.h"
# 1 "/usr/include/features.h" 1 3 4
# 465 "/usr/include/features.h" 3 4
# 1 "/usr/include/sys/cdefs.h" 1 3 4
# 449 "/usr/include/sys/cdefs.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 450 "/usr/include/sys/cdefs.h" 2 3 4
# 1 "/usr/include/bits/long-double.h" 1 3 4
# 451 "/usr/include/sys/cdefs.h" 2 3 4
# 466 "/usr/include/features.h" 2 3 4
# 489 "/usr/include/features.h" 3 4
# 1 "/usr/include/gnu/stubs.h" 1 3 4
# 10 "/usr/include/gnu/stubs.h" 3 4
# 1 "/usr/include/gnu/stubs-64.h" 1 3 4
# 11 "/usr/include/gnu/stubs.h" 2 3 4
# 490 "/usr/include/features.h" 2 3 4
# 209 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_config.h" 2
# 84 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2







# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 1
# 56 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_types.h" 1
# 58 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_types.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 59 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_types.h" 2







enum __attribute__((device_builtin)) cudaRoundMode
{
    cudaRoundNearest,
    cudaRoundZero,
    cudaRoundPosInf,
    cudaRoundMinInf
};
# 57 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 2


# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h" 1
# 58 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 59 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/vector_types.h" 1
# 64 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/vector_types.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 65 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/vector_types.h" 2
# 98 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/vector_types.h"
struct __attribute__((device_builtin)) char1
{
    signed char x;
};

struct __attribute__((device_builtin)) uchar1
{
    unsigned char x;
};


struct __attribute__((device_builtin)) __attribute__((aligned(2))) char2
{
    signed char x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(2))) uchar2
{
    unsigned char x, y;
};

struct __attribute__((device_builtin)) char3
{
    signed char x, y, z;
};

struct __attribute__((device_builtin)) uchar3
{
    unsigned char x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) char4
{
    signed char x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) uchar4
{
    unsigned char x, y, z, w;
};

struct __attribute__((device_builtin)) short1
{
    short x;
};

struct __attribute__((device_builtin)) ushort1
{
    unsigned short x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) short2
{
    short x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) ushort2
{
    unsigned short x, y;
};

struct __attribute__((device_builtin)) short3
{
    short x, y, z;
};

struct __attribute__((device_builtin)) ushort3
{
    unsigned short x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(8))) short4 { short x; short y; short z; short w; };
struct __attribute__((device_builtin)) __attribute__((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; };

struct __attribute__((device_builtin)) int1
{
    int x;
};

struct __attribute__((device_builtin)) uint1
{
    unsigned int x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(8))) int2 { int x; int y; };
struct __attribute__((device_builtin)) __attribute__((aligned(8))) uint2 { unsigned int x; unsigned int y; };

struct __attribute__((device_builtin)) int3
{
    int x, y, z;
};

struct __attribute__((device_builtin)) uint3
{
    unsigned int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) int4
{
    int x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) uint4
{
    unsigned int x, y, z, w;
};

struct __attribute__((device_builtin)) long1
{
    long int x;
};

struct __attribute__((device_builtin)) ulong1
{
    unsigned long x;
};






struct __attribute__((device_builtin)) __attribute__((aligned(2*sizeof(long int)))) long2
{
    long int x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(2*sizeof(unsigned long int)))) ulong2
{
    unsigned long int x, y;
};



struct __attribute__((device_builtin)) long3
{
    long int x, y, z;
};

struct __attribute__((device_builtin)) ulong3
{
    unsigned long int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) long4
{
    long int x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulong4
{
    unsigned long int x, y, z, w;
};

struct __attribute__((device_builtin)) float1
{
    float x;
};
# 274 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/vector_types.h"
struct __attribute__((device_builtin)) __attribute__((aligned(8))) float2 { float x; float y; };




struct __attribute__((device_builtin)) float3
{
    float x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) float4
{
    float x, y, z, w;
};

struct __attribute__((device_builtin)) longlong1
{
    long long int x;
};

struct __attribute__((device_builtin)) ulonglong1
{
    unsigned long long int x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) longlong2
{
    long long int x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulonglong2
{
    unsigned long long int x, y;
};

struct __attribute__((device_builtin)) longlong3
{
    long long int x, y, z;
};

struct __attribute__((device_builtin)) ulonglong3
{
    unsigned long long int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) longlong4
{
    long long int x, y, z ,w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulonglong4
{
    unsigned long long int x, y, z, w;
};

struct __attribute__((device_builtin)) double1
{
    double x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) double2
{
    double x, y;
};

struct __attribute__((device_builtin)) double3
{
    double x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) double4
{
    double x, y, z, w;
};
# 361 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/vector_types.h"
typedef __attribute__((device_builtin)) struct char1 char1;
typedef __attribute__((device_builtin)) struct uchar1 uchar1;
typedef __attribute__((device_builtin)) struct char2 char2;
typedef __attribute__((device_builtin)) struct uchar2 uchar2;
typedef __attribute__((device_builtin)) struct char3 char3;
typedef __attribute__((device_builtin)) struct uchar3 uchar3;
typedef __attribute__((device_builtin)) struct char4 char4;
typedef __attribute__((device_builtin)) struct uchar4 uchar4;
typedef __attribute__((device_builtin)) struct short1 short1;
typedef __attribute__((device_builtin)) struct ushort1 ushort1;
typedef __attribute__((device_builtin)) struct short2 short2;
typedef __attribute__((device_builtin)) struct ushort2 ushort2;
typedef __attribute__((device_builtin)) struct short3 short3;
typedef __attribute__((device_builtin)) struct ushort3 ushort3;
typedef __attribute__((device_builtin)) struct short4 short4;
typedef __attribute__((device_builtin)) struct ushort4 ushort4;
typedef __attribute__((device_builtin)) struct int1 int1;
typedef __attribute__((device_builtin)) struct uint1 uint1;
typedef __attribute__((device_builtin)) struct int2 int2;
typedef __attribute__((device_builtin)) struct uint2 uint2;
typedef __attribute__((device_builtin)) struct int3 int3;
typedef __attribute__((device_builtin)) struct uint3 uint3;
typedef __attribute__((device_builtin)) struct int4 int4;
typedef __attribute__((device_builtin)) struct uint4 uint4;
typedef __attribute__((device_builtin)) struct long1 long1;
typedef __attribute__((device_builtin)) struct ulong1 ulong1;
typedef __attribute__((device_builtin)) struct long2 long2;
typedef __attribute__((device_builtin)) struct ulong2 ulong2;
typedef __attribute__((device_builtin)) struct long3 long3;
typedef __attribute__((device_builtin)) struct ulong3 ulong3;
typedef __attribute__((device_builtin)) struct long4 long4;
typedef __attribute__((device_builtin)) struct ulong4 ulong4;
typedef __attribute__((device_builtin)) struct float1 float1;
typedef __attribute__((device_builtin)) struct float2 float2;
typedef __attribute__((device_builtin)) struct float3 float3;
typedef __attribute__((device_builtin)) struct float4 float4;
typedef __attribute__((device_builtin)) struct longlong1 longlong1;
typedef __attribute__((device_builtin)) struct ulonglong1 ulonglong1;
typedef __attribute__((device_builtin)) struct longlong2 longlong2;
typedef __attribute__((device_builtin)) struct ulonglong2 ulonglong2;
typedef __attribute__((device_builtin)) struct longlong3 longlong3;
typedef __attribute__((device_builtin)) struct ulonglong3 ulonglong3;
typedef __attribute__((device_builtin)) struct longlong4 longlong4;
typedef __attribute__((device_builtin)) struct ulonglong4 ulonglong4;
typedef __attribute__((device_builtin)) struct double1 double1;
typedef __attribute__((device_builtin)) struct double2 double2;
typedef __attribute__((device_builtin)) struct double3 double3;
typedef __attribute__((device_builtin)) struct double4 double4;







struct __attribute__((device_builtin)) dim3
{
    unsigned int x, y, z;


    __attribute__((host)) __attribute__((device)) constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    __attribute__((host)) __attribute__((device)) constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    __attribute__((host)) __attribute__((device)) constexpr operator uint3(void) const { return uint3{x, y, z}; }






};

typedef __attribute__((device_builtin)) struct dim3 dim3;
# 60 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h" 2
# 77 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/limits.h" 1 3 4
# 34 "/usr/lib/gcc/x86_64-redhat-linux/10/include/limits.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/syslimits.h" 1 3 4






# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/limits.h" 1 3 4
# 195 "/usr/lib/gcc/x86_64-redhat-linux/10/include/limits.h" 3 4
# 1 "/usr/include/limits.h" 1 3 4
# 26 "/usr/include/limits.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 27 "/usr/include/limits.h" 2 3 4
# 183 "/usr/include/limits.h" 3 4
# 1 "/usr/include/bits/posix1_lim.h" 1 3 4
# 27 "/usr/include/bits/posix1_lim.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 28 "/usr/include/bits/posix1_lim.h" 2 3 4
# 161 "/usr/include/bits/posix1_lim.h" 3 4
# 1 "/usr/include/bits/local_lim.h" 1 3 4
# 38 "/usr/include/bits/local_lim.h" 3 4
# 1 "/usr/include/linux/limits.h" 1 3 4
# 39 "/usr/include/bits/local_lim.h" 2 3 4
# 162 "/usr/include/bits/posix1_lim.h" 2 3 4
# 184 "/usr/include/limits.h" 2 3 4



# 1 "/usr/include/bits/posix2_lim.h" 1 3 4
# 188 "/usr/include/limits.h" 2 3 4



# 1 "/usr/include/bits/xopen_lim.h" 1 3 4
# 64 "/usr/include/bits/xopen_lim.h" 3 4
# 1 "/usr/include/bits/uio_lim.h" 1 3 4
# 65 "/usr/include/bits/xopen_lim.h" 2 3 4
# 192 "/usr/include/limits.h" 2 3 4
# 196 "/usr/lib/gcc/x86_64-redhat-linux/10/include/limits.h" 2 3 4
# 8 "/usr/lib/gcc/x86_64-redhat-linux/10/include/syslimits.h" 2 3 4
# 35 "/usr/lib/gcc/x86_64-redhat-linux/10/include/limits.h" 2 3 4
# 78 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h" 2
# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 1 3 4
# 143 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 3 4

# 143 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 3 4
typedef long int ptrdiff_t;
# 209 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 3 4
typedef long unsigned int size_t;
# 415 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 3 4
typedef struct {
  long long __max_align_ll __attribute__((__aligned__(__alignof__(long long))));
  long double __max_align_ld __attribute__((__aligned__(__alignof__(long double))));
# 426 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 3 4
} max_align_t;






  typedef decltype(nullptr) nullptr_t;
# 79 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h" 2
# 197 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"

# 197 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
enum __attribute__((device_builtin)) cudaError
{





    cudaSuccess = 0,





    cudaErrorInvalidValue = 1,





    cudaErrorMemoryAllocation = 2,





    cudaErrorInitializationError = 3,






    cudaErrorCudartUnloading = 4,






    cudaErrorProfilerDisabled = 5,







    cudaErrorProfilerNotInitialized = 6,






    cudaErrorProfilerAlreadyStarted = 7,






     cudaErrorProfilerAlreadyStopped = 8,
# 267 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorInvalidConfiguration = 9,





    cudaErrorInvalidPitchValue = 12,





    cudaErrorInvalidSymbol = 13,







    cudaErrorInvalidHostPointer = 16,







    cudaErrorInvalidDevicePointer = 17,





    cudaErrorInvalidTexture = 18,





    cudaErrorInvalidTextureBinding = 19,






    cudaErrorInvalidChannelDescriptor = 20,





    cudaErrorInvalidMemcpyDirection = 21,
# 330 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorAddressOfConstant = 22,
# 339 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorTextureFetchFailed = 23,
# 348 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorTextureNotBound = 24,
# 357 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorSynchronizationError = 25,





    cudaErrorInvalidFilterSetting = 26,





    cudaErrorInvalidNormSetting = 27,







    cudaErrorMixedDeviceExecution = 28,







    cudaErrorNotYetImplemented = 31,
# 394 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorMemoryValueTooLarge = 32,






    cudaErrorStubLibrary = 34,






    cudaErrorInsufficientDriver = 35,






    cudaErrorCallRequiresNewerDriver = 36,





    cudaErrorInvalidSurface = 37,





    cudaErrorDuplicateVariableName = 43,





    cudaErrorDuplicateTextureName = 44,





    cudaErrorDuplicateSurfaceName = 45,
# 449 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorDevicesUnavailable = 46,
# 462 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorIncompatibleDriverContext = 49,





    cudaErrorMissingConfiguration = 52,
# 477 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorPriorLaunchFailure = 53,






    cudaErrorLaunchMaxDepthExceeded = 65,







    cudaErrorLaunchFileScopedTex = 66,







    cudaErrorLaunchFileScopedSurf = 67,
# 515 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorSyncDepthExceeded = 68,
# 527 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorLaunchPendingCountExceeded = 69,





    cudaErrorInvalidDeviceFunction = 98,





    cudaErrorNoDevice = 100,





    cudaErrorInvalidDevice = 101,




    cudaErrorDeviceNotLicensed = 102,
# 559 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
   cudaErrorSoftwareValidityNotEstablished = 103,




    cudaErrorStartupFailure = 127,




    cudaErrorInvalidKernelImage = 200,
# 579 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorDeviceUninitialized = 201,




    cudaErrorMapBufferObjectFailed = 205,




    cudaErrorUnmapBufferObjectFailed = 206,





    cudaErrorArrayIsMapped = 207,




    cudaErrorAlreadyMapped = 208,







    cudaErrorNoKernelImageForDevice = 209,




    cudaErrorAlreadyAcquired = 210,




    cudaErrorNotMapped = 211,





    cudaErrorNotMappedAsArray = 212,





    cudaErrorNotMappedAsPointer = 213,





    cudaErrorECCUncorrectable = 214,





    cudaErrorUnsupportedLimit = 215,





    cudaErrorDeviceAlreadyInUse = 216,





    cudaErrorPeerAccessUnsupported = 217,





    cudaErrorInvalidPtx = 218,




    cudaErrorInvalidGraphicsContext = 219,





    cudaErrorNvlinkUncorrectable = 220,






    cudaErrorJitCompilerNotFound = 221,






    cudaErrorUnsupportedPtxVersion = 222,






    cudaErrorJitCompilationDisabled = 223,




    cudaErrorInvalidSource = 300,




    cudaErrorFileNotFound = 301,




    cudaErrorSharedObjectSymbolNotFound = 302,




    cudaErrorSharedObjectInitFailed = 303,




    cudaErrorOperatingSystem = 304,






    cudaErrorInvalidResourceHandle = 400,





    cudaErrorIllegalState = 401,





    cudaErrorSymbolNotFound = 500,







    cudaErrorNotReady = 600,







    cudaErrorIllegalAddress = 700,
# 761 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorLaunchOutOfResources = 701,
# 772 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorLaunchTimeout = 702,





    cudaErrorLaunchIncompatibleTexturing = 703,






    cudaErrorPeerAccessAlreadyEnabled = 704,






    cudaErrorPeerAccessNotEnabled = 705,
# 805 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorSetOnActiveProcess = 708,






    cudaErrorContextIsDestroyed = 709,






    cudaErrorAssert = 710,






    cudaErrorTooManyPeers = 711,





    cudaErrorHostMemoryAlreadyRegistered = 712,





    cudaErrorHostMemoryNotRegistered = 713,
# 847 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorHardwareStackError = 714,







    cudaErrorIllegalInstruction = 715,
# 864 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorMisalignedAddress = 716,
# 875 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorInvalidAddressSpace = 717,







    cudaErrorInvalidPc = 718,
# 894 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorLaunchFailure = 719,
# 903 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorCooperativeLaunchTooLarge = 720,




    cudaErrorNotPermitted = 800,





    cudaErrorNotSupported = 801,
# 923 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorSystemNotReady = 802,






    cudaErrorSystemDriverMismatch = 803,
# 939 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaErrorCompatNotSupportedOnDevice = 804,




    cudaErrorStreamCaptureUnsupported = 900,





    cudaErrorStreamCaptureInvalidated = 901,





    cudaErrorStreamCaptureMerge = 902,




    cudaErrorStreamCaptureUnmatched = 903,





    cudaErrorStreamCaptureUnjoined = 904,






    cudaErrorStreamCaptureIsolation = 905,





    cudaErrorStreamCaptureImplicit = 906,





    cudaErrorCapturedEvent = 907,






    cudaErrorStreamCaptureWrongThread = 908,




    cudaErrorTimeout = 909,





    cudaErrorGraphExecUpdateFailure = 910,




    cudaErrorUnknown = 999,







    cudaErrorApiFailureBase = 10000
};




enum __attribute__((device_builtin)) cudaChannelFormatKind
{
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
    cudaChannelFormatKindNV12 = 4
};




struct __attribute__((device_builtin)) cudaChannelFormatDesc
{
    int x;
    int y;
    int z;
    int w;
    enum cudaChannelFormatKind f;
};




typedef struct cudaArray *cudaArray_t;




typedef const struct cudaArray *cudaArray_const_t;

struct cudaArray;




typedef struct cudaMipmappedArray *cudaMipmappedArray_t;




typedef const struct cudaMipmappedArray *cudaMipmappedArray_const_t;

struct cudaMipmappedArray;
# 1076 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
struct __attribute__((device_builtin)) cudaArraySparseProperties {
    struct {
        unsigned int width;
        unsigned int height;
        unsigned int depth;
    } tileExtent;
    unsigned int miptailFirstLevel;
    unsigned long long miptailSize;
    unsigned int flags;
    unsigned int reserved[4];
};




enum __attribute__((device_builtin)) cudaMemoryType
{
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
};




enum __attribute__((device_builtin)) cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};






struct __attribute__((device_builtin)) cudaPitchedPtr
{
    void *ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
};






struct __attribute__((device_builtin)) cudaExtent
{
    size_t width;
    size_t height;
    size_t depth;
};






struct __attribute__((device_builtin)) cudaPos
{
    size_t x;
    size_t y;
    size_t z;
};




struct __attribute__((device_builtin)) cudaMemcpy3DParms
{
    cudaArray_t srcArray;
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;

    cudaArray_t dstArray;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;

    struct cudaExtent extent;
    enum cudaMemcpyKind kind;
};




struct __attribute__((device_builtin)) cudaMemcpy3DPeerParms
{
    cudaArray_t srcArray;
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;
    int srcDevice;

    cudaArray_t dstArray;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;
    int dstDevice;

    struct cudaExtent extent;
};




struct __attribute__((device_builtin)) cudaMemsetParams {
    void *dst;
    size_t pitch;
    unsigned int value;
    unsigned int elementSize;
    size_t width;
    size_t height;
};




enum __attribute__((device_builtin)) cudaAccessProperty {
    cudaAccessPropertyNormal = 0,
    cudaAccessPropertyStreaming = 1,
    cudaAccessPropertyPersisting = 2
};
# 1215 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
struct __attribute__((device_builtin)) cudaAccessPolicyWindow {
    void *base_ptr;
    size_t num_bytes;
    float hitRatio;
    enum cudaAccessProperty hitProp;
    enum cudaAccessProperty missProp;
};
# 1233 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
typedef void ( *cudaHostFn_t)(void *userData);




struct __attribute__((device_builtin)) cudaHostNodeParams {
    cudaHostFn_t fn;
    void* userData;
};




enum __attribute__((device_builtin)) cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2

};





enum __attribute__((device_builtin)) cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2
};

enum __attribute__((device_builtin)) cudaSynchronizationPolicy {
    cudaSyncPolicyAuto = 1,
    cudaSyncPolicySpin = 2,
    cudaSyncPolicyYield = 3,
    cudaSyncPolicyBlockingSync = 4
};




enum __attribute__((device_builtin)) cudaStreamAttrID {
    cudaStreamAttributeAccessPolicyWindow = 1,
    cudaStreamAttributeSynchronizationPolicy = 3
};




union __attribute__((device_builtin)) cudaStreamAttrValue {
    struct cudaAccessPolicyWindow accessPolicyWindow;
    enum cudaSynchronizationPolicy syncPolicy;
};




struct cudaGraphicsResource;




enum __attribute__((device_builtin)) cudaGraphicsRegisterFlags
{
    cudaGraphicsRegisterFlagsNone = 0,
    cudaGraphicsRegisterFlagsReadOnly = 1,
    cudaGraphicsRegisterFlagsWriteDiscard = 2,
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
    cudaGraphicsRegisterFlagsTextureGather = 8
};




enum __attribute__((device_builtin)) cudaGraphicsMapFlags
{
    cudaGraphicsMapFlagsNone = 0,
    cudaGraphicsMapFlagsReadOnly = 1,
    cudaGraphicsMapFlagsWriteDiscard = 2
};




enum __attribute__((device_builtin)) cudaGraphicsCubeFace
{
    cudaGraphicsCubeFacePositiveX = 0x00,
    cudaGraphicsCubeFaceNegativeX = 0x01,
    cudaGraphicsCubeFacePositiveY = 0x02,
    cudaGraphicsCubeFaceNegativeY = 0x03,
    cudaGraphicsCubeFacePositiveZ = 0x04,
    cudaGraphicsCubeFaceNegativeZ = 0x05
};




enum __attribute__((device_builtin)) cudaKernelNodeAttrID {
    cudaKernelNodeAttributeAccessPolicyWindow = 1,
    cudaKernelNodeAttributeCooperative = 2
};




union __attribute__((device_builtin)) cudaKernelNodeAttrValue {
    struct cudaAccessPolicyWindow accessPolicyWindow;
    int cooperative;
};




enum __attribute__((device_builtin)) cudaResourceType
{
    cudaResourceTypeArray = 0x00,
    cudaResourceTypeMipmappedArray = 0x01,
    cudaResourceTypeLinear = 0x02,
    cudaResourceTypePitch2D = 0x03
};




enum __attribute__((device_builtin)) cudaResourceViewFormat
{
    cudaResViewFormatNone = 0x00,
    cudaResViewFormatUnsignedChar1 = 0x01,
    cudaResViewFormatUnsignedChar2 = 0x02,
    cudaResViewFormatUnsignedChar4 = 0x03,
    cudaResViewFormatSignedChar1 = 0x04,
    cudaResViewFormatSignedChar2 = 0x05,
    cudaResViewFormatSignedChar4 = 0x06,
    cudaResViewFormatUnsignedShort1 = 0x07,
    cudaResViewFormatUnsignedShort2 = 0x08,
    cudaResViewFormatUnsignedShort4 = 0x09,
    cudaResViewFormatSignedShort1 = 0x0a,
    cudaResViewFormatSignedShort2 = 0x0b,
    cudaResViewFormatSignedShort4 = 0x0c,
    cudaResViewFormatUnsignedInt1 = 0x0d,
    cudaResViewFormatUnsignedInt2 = 0x0e,
    cudaResViewFormatUnsignedInt4 = 0x0f,
    cudaResViewFormatSignedInt1 = 0x10,
    cudaResViewFormatSignedInt2 = 0x11,
    cudaResViewFormatSignedInt4 = 0x12,
    cudaResViewFormatHalf1 = 0x13,
    cudaResViewFormatHalf2 = 0x14,
    cudaResViewFormatHalf4 = 0x15,
    cudaResViewFormatFloat1 = 0x16,
    cudaResViewFormatFloat2 = 0x17,
    cudaResViewFormatFloat4 = 0x18,
    cudaResViewFormatUnsignedBlockCompressed1 = 0x19,
    cudaResViewFormatUnsignedBlockCompressed2 = 0x1a,
    cudaResViewFormatUnsignedBlockCompressed3 = 0x1b,
    cudaResViewFormatUnsignedBlockCompressed4 = 0x1c,
    cudaResViewFormatSignedBlockCompressed4 = 0x1d,
    cudaResViewFormatUnsignedBlockCompressed5 = 0x1e,
    cudaResViewFormatSignedBlockCompressed5 = 0x1f,
    cudaResViewFormatUnsignedBlockCompressed6H = 0x20,
    cudaResViewFormatSignedBlockCompressed6H = 0x21,
    cudaResViewFormatUnsignedBlockCompressed7 = 0x22
};




struct __attribute__((device_builtin)) cudaResourceDesc {
    enum cudaResourceType resType;

    union {
        struct {
            cudaArray_t array;
        } array;
        struct {
            cudaMipmappedArray_t mipmap;
        } mipmap;
        struct {
            void *devPtr;
            struct cudaChannelFormatDesc desc;
            size_t sizeInBytes;
        } linear;
        struct {
            void *devPtr;
            struct cudaChannelFormatDesc desc;
            size_t width;
            size_t height;
            size_t pitchInBytes;
        } pitch2D;
    } res;
};




struct __attribute__((device_builtin)) cudaResourceViewDesc
{
    enum cudaResourceViewFormat format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
};




struct __attribute__((device_builtin)) cudaPointerAttributes
{




    enum cudaMemoryType type;
# 1458 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    int device;





    void *devicePointer;
# 1473 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    void *hostPointer;
};




struct __attribute__((device_builtin)) cudaFuncAttributes
{





   size_t sharedSizeBytes;





   size_t constSizeBytes;




   size_t localSizeBytes;






   int maxThreadsPerBlock;




   int numRegs;






   int ptxVersion;






   int binaryVersion;





   int cacheModeCA;






   int maxDynamicSharedSizeBytes;
# 1545 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
   int preferredShmemCarveout;
};




enum __attribute__((device_builtin)) cudaFuncAttribute
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9,
    cudaFuncAttributeMax
};




enum __attribute__((device_builtin)) cudaFuncCache
{
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3
};





enum __attribute__((device_builtin)) cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2
};




enum __attribute__((device_builtin)) cudaSharedCarveout {
    cudaSharedmemCarveoutDefault = -1,
    cudaSharedmemCarveoutMaxShared = 100,
    cudaSharedmemCarveoutMaxL1 = 0
};




enum __attribute__((device_builtin)) cudaComputeMode
{
    cudaComputeModeDefault = 0,
    cudaComputeModeExclusive = 1,
    cudaComputeModeProhibited = 2,
    cudaComputeModeExclusiveProcess = 3
};




enum __attribute__((device_builtin)) cudaLimit
{
    cudaLimitStackSize = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02,
    cudaLimitDevRuntimeSyncDepth = 0x03,
    cudaLimitDevRuntimePendingLaunchCount = 0x04,
    cudaLimitMaxL2FetchGranularity = 0x05,
    cudaLimitPersistingL2CacheSize = 0x06
};




enum __attribute__((device_builtin)) cudaMemoryAdvise
{
    cudaMemAdviseSetReadMostly = 1,
    cudaMemAdviseUnsetReadMostly = 2,
    cudaMemAdviseSetPreferredLocation = 3,
    cudaMemAdviseUnsetPreferredLocation = 4,
    cudaMemAdviseSetAccessedBy = 5,
    cudaMemAdviseUnsetAccessedBy = 6
};




enum __attribute__((device_builtin)) cudaMemRangeAttribute
{
    cudaMemRangeAttributeReadMostly = 1,
    cudaMemRangeAttributePreferredLocation = 2,
    cudaMemRangeAttributeAccessedBy = 3,
    cudaMemRangeAttributeLastPrefetchLocation = 4
};




enum __attribute__((device_builtin)) cudaOutputMode
{
    cudaKeyValuePair = 0x00,
    cudaCSV = 0x01
};




enum __attribute__((device_builtin)) cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115
};




enum __attribute__((device_builtin)) cudaMemPoolAttr
{
# 1773 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaMemPoolReuseFollowEventDependencies = 0x1,
# 1783 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaMemPoolReuseAllowOpportunistic = 0x2,






    cudaMemPoolReuseAllowInternalDependencies = 0x3,
# 1799 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    cudaMemPoolAttrReleaseThreshold = 0x4
};

enum __attribute__((device_builtin)) cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1
};

struct __attribute__((device_builtin)) cudaMemLocation {
    enum cudaMemLocationType type;
    int id;
};

enum __attribute__((device_builtin)) cudaMemAccessFlags {
    cudaMemAccessFlagsProtNone = 0,
    cudaMemAccessFlagsProtRead = 1,
    cudaMemAccessFlagsProtReadWrite = 3
};

struct __attribute__((device_builtin)) cudaMemAccessDesc {
    struct cudaMemLocation location;
    enum cudaMemAccessFlags flags;
};

enum __attribute__((device_builtin)) cudaMemAllocationType {
    cudaMemAllocationTypeInvalid = 0x0,
    cudaMemAllocationTypePinned = 0x1,
    cudaMemAllocationTypeMax = 0xFFFFFFFF
};

enum __attribute__((device_builtin)) cudaMemAllocationHandleType {
    cudaMemHandleTypeNone = 0x0,
    cudaMemHandleTypePosixFileDescriptor = 0x1,
    cudaMemHandleTypeWin32 = 0x2,
    cudaMemHandleTypeWin32Kmt = 0x4
};

struct __attribute__((device_builtin)) cudaMemPoolProps {
    enum cudaMemAllocationType allocType;
    enum cudaMemAllocationHandleType handleTypes;
    struct cudaMemLocation location;
    void *win32SecurityAttributes;
    unsigned char reserved[64];
};

struct __attribute__((device_builtin)) cudaMemPoolPtrExportData {
    unsigned char reserved[64];
};





enum __attribute__((device_builtin)) cudaDeviceP2PAttr {
    cudaDevP2PAttrPerformanceRank = 1,
    cudaDevP2PAttrAccessSupported = 2,
    cudaDevP2PAttrNativeAtomicSupported = 3,
    cudaDevP2PAttrCudaArrayAccessSupported = 4
};






struct __attribute__((device_builtin)) CUuuid_st {
    char bytes[16];
};
typedef __attribute__((device_builtin)) struct CUuuid_st CUuuid;

typedef __attribute__((device_builtin)) struct CUuuid_st cudaUUID_t;




struct __attribute__((device_builtin)) cudaDeviceProp
{
    char name[256];
    cudaUUID_t uuid;
    char luid[8];
    unsigned int luidDeviceNodeMask;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
};
# 2049 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
typedef __attribute__((device_builtin)) struct __attribute__((device_builtin)) cudaIpcEventHandle_st
{
    char reserved[64];
}cudaIpcEventHandle_t;




typedef __attribute__((device_builtin)) struct __attribute__((device_builtin)) cudaIpcMemHandle_st
{
    char reserved[64];
}cudaIpcMemHandle_t;




enum __attribute__((device_builtin)) cudaExternalMemoryHandleType {



    cudaExternalMemoryHandleTypeOpaqueFd = 1,



    cudaExternalMemoryHandleTypeOpaqueWin32 = 2,



    cudaExternalMemoryHandleTypeOpaqueWin32Kmt = 3,



    cudaExternalMemoryHandleTypeD3D12Heap = 4,



    cudaExternalMemoryHandleTypeD3D12Resource = 5,



    cudaExternalMemoryHandleTypeD3D11Resource = 6,



    cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,



    cudaExternalMemoryHandleTypeNvSciBuf = 8
};
# 2140 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
struct __attribute__((device_builtin)) cudaExternalMemoryHandleDesc {



    enum cudaExternalMemoryHandleType type;
    union {





        int fd;
# 2167 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
        struct {



            void *handle;




            const void *name;
        } win32;




        const void *nvSciBufObject;
    } handle;



    unsigned long long size;



    unsigned int flags;
};




struct __attribute__((device_builtin)) cudaExternalMemoryBufferDesc {



    unsigned long long offset;



    unsigned long long size;



    unsigned int flags;
};




struct __attribute__((device_builtin)) cudaExternalMemoryMipmappedArrayDesc {




    unsigned long long offset;



    struct cudaChannelFormatDesc formatDesc;



    struct cudaExtent extent;




    unsigned int flags;



    unsigned int numLevels;
};




enum __attribute__((device_builtin)) cudaExternalSemaphoreHandleType {



    cudaExternalSemaphoreHandleTypeOpaqueFd = 1,



    cudaExternalSemaphoreHandleTypeOpaqueWin32 = 2,



    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,



    cudaExternalSemaphoreHandleTypeD3D12Fence = 4,



    cudaExternalSemaphoreHandleTypeD3D11Fence = 5,



     cudaExternalSemaphoreHandleTypeNvSciSync = 6,



    cudaExternalSemaphoreHandleTypeKeyedMutex = 7,



    cudaExternalSemaphoreHandleTypeKeyedMutexKmt = 8,



    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9,



    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10
};




struct __attribute__((device_builtin)) cudaExternalSemaphoreHandleDesc {



    enum cudaExternalSemaphoreHandleType type;
    union {






        int fd;
# 2317 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
        struct {



            void *handle;




            const void *name;
        } win32;



        const void* nvSciSyncObj;
    } handle;



    unsigned int flags;
};
# 2438 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
struct __attribute__((device_builtin)) cudaExternalSemaphoreSignalParams{
    struct {



        struct {



            unsigned long long value;
        } fence;
        union {




            void *fence;
            unsigned long long reserved;
        } nvSciSync;



        struct {



            unsigned long long key;
        } keyedMutex;
        unsigned int reserved[12];
    } params;
# 2478 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    unsigned int flags;
    unsigned int reserved[16];
};




struct __attribute__((device_builtin)) cudaExternalSemaphoreWaitParams {
    struct {



        struct {



            unsigned long long value;
        } fence;
        union {




            void *fence;
            unsigned long long reserved;
        } nvSciSync;



        struct {



            unsigned long long key;



            unsigned int timeoutMs;
        } keyedMutex;
        unsigned int reserved[10];
    } params;
# 2529 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
    unsigned int flags;
    unsigned int reserved[16];
};
# 2543 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_types.h"
typedef __attribute__((device_builtin)) enum cudaError cudaError_t;




typedef __attribute__((device_builtin)) struct CUstream_st *cudaStream_t;




typedef __attribute__((device_builtin)) struct CUevent_st *cudaEvent_t;




typedef __attribute__((device_builtin)) struct cudaGraphicsResource *cudaGraphicsResource_t;




typedef __attribute__((device_builtin)) enum cudaOutputMode cudaOutputMode_t;




typedef __attribute__((device_builtin)) struct CUexternalMemory_st *cudaExternalMemory_t;




typedef __attribute__((device_builtin)) struct CUexternalSemaphore_st *cudaExternalSemaphore_t;




typedef __attribute__((device_builtin)) struct CUgraph_st *cudaGraph_t;




typedef __attribute__((device_builtin)) struct CUgraphNode_st *cudaGraphNode_t;




typedef __attribute__((device_builtin)) struct CUfunc_st *cudaFunction_t;




typedef __attribute__((device_builtin)) struct CUmemPoolHandle_st *cudaMemPool_t;




enum __attribute__((device_builtin)) cudaCGScope {
    cudaCGScopeInvalid = 0,
    cudaCGScopeGrid = 1,
    cudaCGScopeMultiGrid = 2
};




struct __attribute__((device_builtin)) cudaLaunchParams
{
    void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;
};




struct __attribute__((device_builtin)) cudaKernelNodeParams {
    void* func;
    dim3 gridDim;
    dim3 blockDim;
    unsigned int sharedMemBytes;
    void **kernelParams;
    void **extra;
};




struct __attribute__((device_builtin)) cudaExternalSemaphoreSignalNodeParams {
    cudaExternalSemaphore_t* extSemArray;
    const struct cudaExternalSemaphoreSignalParams* paramsArray;
    unsigned int numExtSems;
};




struct __attribute__((device_builtin)) cudaExternalSemaphoreWaitNodeParams {
    cudaExternalSemaphore_t* extSemArray;
    const struct cudaExternalSemaphoreWaitParams* paramsArray;
    unsigned int numExtSems;
};




enum __attribute__((device_builtin)) cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0x00,
    cudaGraphNodeTypeMemcpy = 0x01,
    cudaGraphNodeTypeMemset = 0x02,
    cudaGraphNodeTypeHost = 0x03,
    cudaGraphNodeTypeGraph = 0x04,
    cudaGraphNodeTypeEmpty = 0x05,
    cudaGraphNodeTypeWaitEvent = 0x06,
    cudaGraphNodeTypeEventRecord = 0x07,
    cudaGraphNodeTypeCount
};




typedef struct CUgraphExec_st* cudaGraphExec_t;




enum __attribute__((device_builtin)) cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess = 0x0,
    cudaGraphExecUpdateError = 0x1,
    cudaGraphExecUpdateErrorTopologyChanged = 0x2,
    cudaGraphExecUpdateErrorNodeTypeChanged = 0x3,
    cudaGraphExecUpdateErrorFunctionChanged = 0x4,
    cudaGraphExecUpdateErrorParametersChanged = 0x5,
    cudaGraphExecUpdateErrorNotSupported = 0x6,
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 0x7
};
# 60 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 2


# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/surface_types.h" 1
# 84 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/surface_types.h"
enum __attribute__((device_builtin)) cudaSurfaceBoundaryMode
{
    cudaBoundaryModeZero = 0,
    cudaBoundaryModeClamp = 1,
    cudaBoundaryModeTrap = 2
};




enum __attribute__((device_builtin)) cudaSurfaceFormatMode
{
    cudaFormatModeForced = 0,
    cudaFormatModeAuto = 1
};




struct __attribute__((device_builtin)) surfaceReference
{



    struct cudaChannelFormatDesc channelDesc;
};




typedef __attribute__((device_builtin)) unsigned long long cudaSurfaceObject_t;
# 63 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_types.h" 1
# 84 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_types.h"
enum __attribute__((device_builtin)) cudaTextureAddressMode
{
    cudaAddressModeWrap = 0,
    cudaAddressModeClamp = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3
};




enum __attribute__((device_builtin)) cudaTextureFilterMode
{
    cudaFilterModePoint = 0,
    cudaFilterModeLinear = 1
};




enum __attribute__((device_builtin)) cudaTextureReadMode
{
    cudaReadModeElementType = 0,
    cudaReadModeNormalizedFloat = 1
};




struct __attribute__((device_builtin)) textureReference
{



    int normalized;



    enum cudaTextureFilterMode filterMode;



    enum cudaTextureAddressMode addressMode[3];



    struct cudaChannelFormatDesc channelDesc;



    int sRGB;



    unsigned int maxAnisotropy;



    enum cudaTextureFilterMode mipmapFilterMode;



    float mipmapLevelBias;



    float minMipmapLevelClamp;



    float maxMipmapLevelClamp;



    int disableTrilinearOptimization;
    int __cudaReserved[14];
};




struct __attribute__((device_builtin)) cudaTextureDesc
{



    enum cudaTextureAddressMode addressMode[3];



    enum cudaTextureFilterMode filterMode;



    enum cudaTextureReadMode readMode;



    int sRGB;



    float borderColor[4];



    int normalizedCoords;



    unsigned int maxAnisotropy;



    enum cudaTextureFilterMode mipmapFilterMode;



    float mipmapLevelBias;



    float minMipmapLevelClamp;



    float maxMipmapLevelClamp;



    int disableTrilinearOptimization;
};




typedef __attribute__((device_builtin)) unsigned long long cudaTextureObject_t;
# 64 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 2
# 92 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/library_types.h" 1
# 54 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/library_types.h"
typedef enum cudaDataType_t
{
 CUDA_R_16F = 2,
 CUDA_C_16F = 6,
 CUDA_R_16BF = 14,
 CUDA_C_16BF = 15,
 CUDA_R_32F = 0,
 CUDA_C_32F = 4,
 CUDA_R_64F = 1,
 CUDA_C_64F = 5,
 CUDA_R_4I = 16,
 CUDA_C_4I = 17,
 CUDA_R_4U = 18,
 CUDA_C_4U = 19,
 CUDA_R_8I = 3,
 CUDA_C_8I = 7,
 CUDA_R_8U = 8,
 CUDA_C_8U = 9,
 CUDA_R_16I = 20,
 CUDA_C_16I = 21,
 CUDA_R_16U = 22,
 CUDA_C_16U = 23,
 CUDA_R_32I = 10,
 CUDA_C_32I = 11,
 CUDA_R_32U = 12,
 CUDA_C_32U = 13,
 CUDA_R_64I = 24,
 CUDA_C_64I = 25,
 CUDA_R_64U = 26,
 CUDA_C_64U = 27
} cudaDataType;


typedef enum libraryPropertyType_t
{
 MAJOR_VERSION,
 MINOR_VERSION,
 PATCH_LEVEL
} libraryPropertyType;
# 93 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2


# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/channel_descriptor.h" 1
# 61 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/channel_descriptor.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h" 1
# 146 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 147 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 1
# 148 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h" 2

# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h" 1
# 64 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
extern "C" {


struct cudaFuncAttributes;







__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaMalloc(void **p, size_t s)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *p, const void *c)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaGetDevice(int *device)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags)
{
  return cudaErrorUnknown;
}




}
# 119 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 120 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h" 2

extern "C"
{
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceSynchronize(void);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaGetLastError(void);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaPeekAtLastError(void);
extern __attribute__((device)) __attribute__((cudart_builtin)) const char* cudaGetErrorString(cudaError_t error);
extern __attribute__((device)) __attribute__((cudart_builtin)) const char* cudaGetErrorName(cudaError_t error);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaGetDeviceCount(int *count);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaGetDevice(int *device);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaStreamDestroy(cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaEventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaEventDestroy(cudaEvent_t event);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaFree(void *devPtr);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMalloc(void **devPtr, size_t size);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpyAsync_ptsz(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy2DAsync_ptsz(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy3DAsync_ptsz(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemsetAsync_ptsz(void *devPtr, int value, size_t count, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemset2DAsync_ptsz(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMemset3DAsync_ptsz(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
# 180 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
extern __attribute__((device)) __attribute__((cudart_builtin)) void * cudaGetParameterBuffer(size_t alignment, size_t size);
# 208 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
extern __attribute__((device)) __attribute__((cudart_builtin)) void * cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaLaunchDevice_ptsz(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaLaunchDeviceV2_ptsz(void *parameterBuffer, cudaStream_t stream);
# 228 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
    extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaLaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream);
    extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream);


extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags);

extern __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long cudaCGGetIntrinsicHandle(enum cudaCGScope scope);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaCGSynchronize(unsigned long long handle, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaCGSynchronizeGrid(unsigned long long handle, unsigned int flags);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaCGGetSize(unsigned int *numThreads, unsigned int *numGrids, unsigned long long handle);
extern __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaCGGetRank(unsigned int *threadRank, unsigned int *gridRank, unsigned long long handle);
}

template <typename T> static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaMalloc(T **devPtr, size_t size);
template <typename T> static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, T *entry);
template <typename T> static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, T func, int blockSize, size_t dynamicSmemSize);
template <typename T> static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned int flags);
# 150 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h" 2
# 262 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern "C" {
# 297 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceReset(void);
# 318 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceSynchronize(void);
# 405 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
# 440 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
# 463 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc, int device);
# 497 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
# 534 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
# 578 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);
# 609 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
# 653 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);
# 680 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);
# 710 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);
# 758 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event);
# 799 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle);
# 842 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr);
# 906 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags);
# 942 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaIpcCloseMemHandle(void *devPtr);
# 985 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaThreadExit(void);
# 1011 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaThreadSynchronize(void);
# 1060 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value);
# 1093 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit);
# 1129 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig);
# 1176 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig);
# 1239 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaGetLastError(void);
# 1287 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaPeekAtLastError(void);
# 1303 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) const char* cudaGetErrorName(cudaError_t error);
# 1319 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) const char* cudaGetErrorString(cudaError_t error);
# 1347 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaGetDeviceCount(int *count);
# 1625 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
# 1818 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
# 1836 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device);
# 1860 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool);
# 1880 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device);
# 1928 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, int device, int flags);
# 1968 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
# 1989 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
# 2026 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaSetDevice(int device);
# 2047 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaGetDevice(int *device);
# 2078 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaSetValidDevices(int *device_arr, int len);
# 2147 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaSetDeviceFlags( unsigned int flags );
# 2193 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetDeviceFlags( unsigned int *flags );
# 2233 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamCreate(cudaStream_t *pStream);
# 2265 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
# 2311 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority);
# 2338 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority);
# 2363 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags);
# 2378 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaCtxResetPersistingL2Cache(void);
# 2398 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src);
# 2419 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamGetAttribute(
        cudaStream_t hStream, enum cudaStreamAttrID attr,
        union cudaStreamAttrValue *value_out);
# 2443 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamSetAttribute(
        cudaStream_t hStream, enum cudaStreamAttrID attr,
        const union cudaStreamAttrValue *value);
# 2477 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamDestroy(cudaStream_t stream);
# 2508 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0);







typedef void ( *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void *userData);
# 2583 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamAddCallback(cudaStream_t stream,
        cudaStreamCallback_t callback, void *userData, unsigned int flags);
# 2607 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamSynchronize(cudaStream_t stream);
# 2632 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamQuery(cudaStream_t stream);
# 2716 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length = 0, unsigned int flags = 0x04);
# 2755 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);
# 2806 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode);
# 2834 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph);
# 2872 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus);
# 2900 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus, unsigned long long *pId);
# 2937 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaEventCreate(cudaEvent_t *event);
# 2974 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
# 3014 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
# 3061 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned int flags = 0);
# 3093 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaEventQuery(cudaEvent_t event);
# 3123 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaEventSynchronize(cudaEvent_t event);
# 3152 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaEventDestroy(cudaEvent_t event);
# 3195 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
# 3372 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaImportExternalMemory(cudaExternalMemory_t *extMem_out, const struct cudaExternalMemoryHandleDesc *memHandleDesc);
# 3426 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaExternalMemoryGetMappedBuffer(void **devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc *bufferDesc);
# 3485 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc);
# 3509 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem);
# 3662 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out, const struct cudaExternalSemaphoreHandleDesc *semHandleDesc);
# 3729 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, cudaStream_t stream = 0);
# 3805 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, cudaStream_t stream = 0);
# 3828 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem);
# 3895 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
# 3952 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
# 4051 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags = 0);
# 4098 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig);
# 4153 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config);
# 4186 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
# 4223 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value);
# 4249 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaSetDoubleForDevice(double *d);
# 4273 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaSetDoubleForHost(double *d);
# 4341 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);
# 4398 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);
# 4427 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, const void *func, int numBlocks, int blockSize);
# 4472 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
# 4593 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags = 0x01);
# 4626 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMalloc(void **devPtr, size_t size);
# 4659 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMallocHost(void **ptr, size_t size);
# 4702 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
# 4751 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height = 0, unsigned int flags = 0);
# 4780 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaFree(void *devPtr);
# 4803 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaFreeHost(void *ptr);
# 4826 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaFreeArray(cudaArray_t array);
# 4849 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
# 4915 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
# 5008 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
# 5031 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaHostUnregister(void *ptr);
# 5076 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
# 5098 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost);
# 5137 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
# 5279 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags = 0);
# 5421 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags = 0);
# 5454 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level);
# 5559 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
# 5590 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p);
# 5708 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream = 0);
# 5734 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream = 0);
# 5756 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
# 5782 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array);
# 5811 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaArrayGetPlane(cudaArray_t *pPlaneArray, cudaArray_t hArray, unsigned int planeIdx);
# 5839 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaArray_t array);
# 5869 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaMipmappedArray_t mipmap);
# 5914 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
# 5949 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
# 5998 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
# 6048 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
# 6098 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
# 6145 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice);
# 6188 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset = 0, enum cudaMemcpyKind kind = cudaMemcpyHostToDevice);
# 6231 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset = 0, enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
# 6288 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 6323 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream = 0);
# 6386 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 6444 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 6501 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 6552 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 6603 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 6632 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemset(void *devPtr, int value, size_t count);
# 6666 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
# 6712 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
# 6748 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream = 0);
# 6789 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0);
# 6842 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream = 0);
# 6870 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol);
# 6897 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol);
# 6967 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream = 0);
# 7083 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device);
# 7142 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void *devPtr, size_t count);
# 7181 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemRangeGetAttributes(void **data, size_t *dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count);
# 7241 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
# 7283 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
# 7326 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice);
# 7377 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 7427 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = 0);
# 7491 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream);
# 7514 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream);
# 7539 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep);
# 7576 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value );
# 7613 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value );
# 7628 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc *descList, size_t count);
# 7641 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags *flags, cudaMemPool_t memPool, struct cudaMemLocation *location);
# 7660 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPoolCreate(cudaMemPool_t *memPool, const struct cudaMemPoolProps *poolProps);
# 7682 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool);
# 7712 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream);
# 7737 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPoolExportToShareableHandle(
    void *shareableHandle,
    cudaMemPool_t memPool,
    enum cudaMemAllocationHandleType handleType,
    unsigned int flags);
# 7764 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPoolImportFromShareableHandle(
    cudaMemPool_t *memPool,
    void *shareableHandle,
    enum cudaMemAllocationHandleType handleType,
    unsigned int flags);
# 7787 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData *exportData, void *ptr);
# 7816 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaMemPoolImportPointer(void **ptr, cudaMemPool_t memPool, struct cudaMemPoolPtrExportData *exportData);
# 7968 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr);
# 8009 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice);
# 8051 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
# 8073 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
# 8137 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);
# 8172 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags);
# 8211 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream = 0);
# 8246 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream = 0);
# 8278 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource);
# 8316 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);
# 8345 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource);
# 8416 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size = 
# 8416 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h" 3 4
                                                                                                                                                                                                        (0x7fffffff * 2U + 1U)
# 8416 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
                                                                                                                                                                                                                      );
# 8475 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch);
# 8513 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc);
# 8553 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaBindTextureToMipmappedArray(const struct textureReference *texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc *desc);
# 8579 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaUnbindTexture(const struct textureReference *texref);
# 8608 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);
# 8638 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaGetTextureReference(const struct textureReference **texref, const void *symbol);
# 8683 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaBindSurfaceToArray(const struct surfaceReference *surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc);
# 8708 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((deprecated)) __attribute__((host)) cudaError_t cudaGetSurfaceReference(const struct surfaceReference **surfref, const void *symbol);
# 8743 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array);
# 8773 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);
# 8991 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc, const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc);
# 9011 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
# 9031 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaTextureObject_t texObject);
# 9051 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject);
# 9072 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject);
# 9117 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const struct cudaResourceDesc *pResDesc);
# 9137 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);
# 9156 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject);
# 9190 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaDriverGetVersion(int *driverVersion);
# 9215 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) __attribute__((cudart_builtin)) cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
# 9262 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags);
# 9359 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaKernelNodeParams *pNodeParams);
# 9392 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams *pNodeParams);
# 9417 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams);
# 9437 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphKernelNodeCopyAttributes(
        cudaGraphNode_t hSrc,
        cudaGraphNode_t hDst);
# 9460 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphKernelNodeGetAttribute(
    cudaGraphNode_t hNode,
    enum cudaKernelNodeAttrID attr,
    union cudaKernelNodeAttrValue *value_out);
# 9484 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphKernelNodeSetAttribute(
    cudaGraphNode_t hNode,
    enum cudaKernelNodeAttrID attr,
    const union cudaKernelNodeAttrValue *value);
# 9534 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams);
# 9593 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphAddMemcpyNodeToSymbol(
    cudaGraphNode_t *pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies,
    size_t numDependencies,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
# 9662 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphAddMemcpyNodeFromSymbol(
    cudaGraphNode_t* pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies,
    size_t numDependencies,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
# 9730 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphAddMemcpyNode1D(
    cudaGraphNode_t *pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies,
    size_t numDependencies,
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind);
# 9762 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms *pNodeParams);
# 9788 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams);
# 9827 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(
    cudaGraphNode_t node,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
# 9873 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(
    cudaGraphNode_t node,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
# 9919 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphMemcpyNodeSetParams1D(
    cudaGraphNode_t node,
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind);
# 9966 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemsetParams *pMemsetParams);
# 9989 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams *pNodeParams);
# 10012 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams);
# 10053 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaHostNodeParams *pNodeParams);
# 10076 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams *pNodeParams);
# 10099 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams);
# 10137 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph);
# 10161 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph);
# 10198 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies);
# 10242 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event);
# 10269 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out);
# 10296 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event);
# 10343 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event);
# 10370 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out);
# 10397 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event);
# 10443 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
# 10476 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreSignalNodeParams *params_out);
# 10503 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
# 10549 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
# 10582 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreWaitNodeParams *params_out);
# 10609 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
# 10637 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph);
# 10665 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph);
# 10696 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType *pType);
# 10727 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes);
# 10758 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes);
# 10792 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges);
# 10823 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies);
# 10855 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes);
# 10886 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies);
# 10917 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies);
# 10944 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node);
# 10981 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize);
# 11015 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams);
# 11059 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams);
# 11108 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
# 11165 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
# 11220 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphExecMemcpyNodeSetParams1D(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind);
# 11268 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams);
# 11301 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams);
# 11343 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph);
# 11378 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event);
# 11413 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event);
# 11451 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
# 11489 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
# 11564 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out, enum cudaGraphExecUpdateResult *updateResult_out);
# 11588 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
 extern __attribute__((host)) cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream);
# 11615 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
# 11638 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
# 11659 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGraphDestroy(cudaGraph_t graph);




extern __attribute__((host)) cudaError_t cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId);
# 11840 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern __attribute__((host)) cudaError_t cudaGetFuncBySymbol(cudaFunction_t* functionPtr, const void* symbolPtr);
# 11977 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
}
# 62 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/channel_descriptor.h" 2
# 104 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/channel_descriptor.h"
template<class T> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc(void)
{
  return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
}

static __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDescHalf(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

static __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDescHalf1(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

static __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDescHalf2(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
}

static __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDescHalf4(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<char>(void)
{
  int e = (int)sizeof(char) * 8;




  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);

}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<signed char>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<unsigned char>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<char1>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uchar1>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<char2>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uchar2>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<char4>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uchar4>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<short>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<unsigned short>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<short1>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<ushort1>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<short2>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<ushort2>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<short4>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<ushort4>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<int>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<unsigned int>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<int1>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uint1>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<int2>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uint2>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<int4>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<uint4>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}
# 376 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/channel_descriptor.h"
template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<float>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<float1>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<float2>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDesc<float4>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}

static __inline__ __attribute__((host)) cudaChannelFormatDesc cudaCreateChannelDescNV12(void)
{
    int e = (int)sizeof(char) * 8;

    return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindNV12);
}
# 96 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2

# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_functions.h" 1
# 53 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 1
# 54 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 55 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_functions.h" 2
# 79 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_functions.h"
static __inline__ __attribute__((host)) struct cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz)
{
  struct cudaPitchedPtr s;

  s.ptr = d;
  s.pitch = p;
  s.xsize = xsz;
  s.ysize = ysz;

  return s;
}
# 106 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_functions.h"
static __inline__ __attribute__((host)) struct cudaPos make_cudaPos(size_t x, size_t y, size_t z)
{
  struct cudaPos p;

  p.x = x;
  p.y = y;
  p.z = z;

  return p;
}
# 132 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/driver_functions.h"
static __inline__ __attribute__((host)) struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d)
{
  struct cudaExtent e;

  e.width = w;
  e.height = h;
  e.depth = d;

  return e;
}
# 98 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2


# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 101 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/vector_functions.h" 1
# 73 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/vector_functions.h"
static __inline__ __attribute__((host)) __attribute__((device)) char1 make_char1(signed char x);

static __inline__ __attribute__((host)) __attribute__((device)) uchar1 make_uchar1(unsigned char x);

static __inline__ __attribute__((host)) __attribute__((device)) char2 make_char2(signed char x, signed char y);

static __inline__ __attribute__((host)) __attribute__((device)) uchar2 make_uchar2(unsigned char x, unsigned char y);

static __inline__ __attribute__((host)) __attribute__((device)) char3 make_char3(signed char x, signed char y, signed char z);

static __inline__ __attribute__((host)) __attribute__((device)) uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z);

static __inline__ __attribute__((host)) __attribute__((device)) char4 make_char4(signed char x, signed char y, signed char z, signed char w);

static __inline__ __attribute__((host)) __attribute__((device)) uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w);

static __inline__ __attribute__((host)) __attribute__((device)) short1 make_short1(short x);

static __inline__ __attribute__((host)) __attribute__((device)) ushort1 make_ushort1(unsigned short x);

static __inline__ __attribute__((host)) __attribute__((device)) short2 make_short2(short x, short y);

static __inline__ __attribute__((host)) __attribute__((device)) ushort2 make_ushort2(unsigned short x, unsigned short y);

static __inline__ __attribute__((host)) __attribute__((device)) short3 make_short3(short x,short y, short z);

static __inline__ __attribute__((host)) __attribute__((device)) ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z);

static __inline__ __attribute__((host)) __attribute__((device)) short4 make_short4(short x, short y, short z, short w);

static __inline__ __attribute__((host)) __attribute__((device)) ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w);

static __inline__ __attribute__((host)) __attribute__((device)) int1 make_int1(int x);

static __inline__ __attribute__((host)) __attribute__((device)) uint1 make_uint1(unsigned int x);

static __inline__ __attribute__((host)) __attribute__((device)) int2 make_int2(int x, int y);

static __inline__ __attribute__((host)) __attribute__((device)) uint2 make_uint2(unsigned int x, unsigned int y);

static __inline__ __attribute__((host)) __attribute__((device)) int3 make_int3(int x, int y, int z);

static __inline__ __attribute__((host)) __attribute__((device)) uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z);

static __inline__ __attribute__((host)) __attribute__((device)) int4 make_int4(int x, int y, int z, int w);

static __inline__ __attribute__((host)) __attribute__((device)) uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w);

static __inline__ __attribute__((host)) __attribute__((device)) long1 make_long1(long int x);

static __inline__ __attribute__((host)) __attribute__((device)) ulong1 make_ulong1(unsigned long int x);

static __inline__ __attribute__((host)) __attribute__((device)) long2 make_long2(long int x, long int y);

static __inline__ __attribute__((host)) __attribute__((device)) ulong2 make_ulong2(unsigned long int x, unsigned long int y);

static __inline__ __attribute__((host)) __attribute__((device)) long3 make_long3(long int x, long int y, long int z);

static __inline__ __attribute__((host)) __attribute__((device)) ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z);

static __inline__ __attribute__((host)) __attribute__((device)) long4 make_long4(long int x, long int y, long int z, long int w);

static __inline__ __attribute__((host)) __attribute__((device)) ulong4 make_ulong4(unsigned long int x, unsigned long int y, unsigned long int z, unsigned long int w);

static __inline__ __attribute__((host)) __attribute__((device)) float1 make_float1(float x);

static __inline__ __attribute__((host)) __attribute__((device)) float2 make_float2(float x, float y);

static __inline__ __attribute__((host)) __attribute__((device)) float3 make_float3(float x, float y, float z);

static __inline__ __attribute__((host)) __attribute__((device)) float4 make_float4(float x, float y, float z, float w);

static __inline__ __attribute__((host)) __attribute__((device)) longlong1 make_longlong1(long long int x);

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong1 make_ulonglong1(unsigned long long int x);

static __inline__ __attribute__((host)) __attribute__((device)) longlong2 make_longlong2(long long int x, long long int y);

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y);

static __inline__ __attribute__((host)) __attribute__((device)) longlong3 make_longlong3(long long int x, long long int y, long long int z);

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong3 make_ulonglong3(unsigned long long int x, unsigned long long int y, unsigned long long int z);

static __inline__ __attribute__((host)) __attribute__((device)) longlong4 make_longlong4(long long int x, long long int y, long long int z, long long int w);

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong4 make_ulonglong4(unsigned long long int x, unsigned long long int y, unsigned long long int z, unsigned long long int w);

static __inline__ __attribute__((host)) __attribute__((device)) double1 make_double1(double x);

static __inline__ __attribute__((host)) __attribute__((device)) double2 make_double2(double x, double y);

static __inline__ __attribute__((host)) __attribute__((device)) double3 make_double3(double x, double y, double z);

static __inline__ __attribute__((host)) __attribute__((device)) double4 make_double4(double x, double y, double z, double w);




# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/vector_functions.hpp" 1
# 73 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/vector_functions.hpp"
static __inline__ __attribute__((host)) __attribute__((device)) char1 make_char1(signed char x)
{
  char1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uchar1 make_uchar1(unsigned char x)
{
  uchar1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) char2 make_char2(signed char x, signed char y)
{
  char2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uchar2 make_uchar2(unsigned char x, unsigned char y)
{
  uchar2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) char3 make_char3(signed char x, signed char y, signed char z)
{
  char3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
{
  uchar3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) char4 make_char4(signed char x, signed char y, signed char z, signed char w)
{
  char4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
  uchar4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) short1 make_short1(short x)
{
  short1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ushort1 make_ushort1(unsigned short x)
{
  ushort1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) short2 make_short2(short x, short y)
{
  short2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ushort2 make_ushort2(unsigned short x, unsigned short y)
{
  ushort2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) short3 make_short3(short x,short y, short z)
{
  short3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z)
{
  ushort3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) short4 make_short4(short x, short y, short z, short w)
{
  short4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w)
{
  ushort4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) int1 make_int1(int x)
{
  int1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uint1 make_uint1(unsigned int x)
{
  uint1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) int2 make_int2(int x, int y)
{
  int2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uint2 make_uint2(unsigned int x, unsigned int y)
{
  uint2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) int3 make_int3(int x, int y, int z)
{
  int3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
  uint3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) int4 make_int4(int x, int y, int z, int w)
{
  int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
  uint4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) long1 make_long1(long int x)
{
  long1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulong1 make_ulong1(unsigned long int x)
{
  ulong1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) long2 make_long2(long int x, long int y)
{
  long2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulong2 make_ulong2(unsigned long int x, unsigned long int y)
{
  ulong2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) long3 make_long3(long int x, long int y, long int z)
{
  long3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z)
{
  ulong3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) long4 make_long4(long int x, long int y, long int z, long int w)
{
  long4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulong4 make_ulong4(unsigned long int x, unsigned long int y, unsigned long int z, unsigned long int w)
{
  ulong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) float1 make_float1(float x)
{
  float1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) float2 make_float2(float x, float y)
{
  float2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) float3 make_float3(float x, float y, float z)
{
  float3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) float4 make_float4(float x, float y, float z, float w)
{
  float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) longlong1 make_longlong1(long long int x)
{
  longlong1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong1 make_ulonglong1(unsigned long long int x)
{
  ulonglong1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) longlong2 make_longlong2(long long int x, long long int y)
{
  longlong2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y)
{
  ulonglong2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) longlong3 make_longlong3(long long int x, long long int y, long long int z)
{
  longlong3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong3 make_ulonglong3(unsigned long long int x, unsigned long long int y, unsigned long long int z)
{
  ulonglong3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) longlong4 make_longlong4(long long int x, long long int y, long long int z, long long int w)
{
  longlong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) ulonglong4 make_ulonglong4(unsigned long long int x, unsigned long long int y, unsigned long long int z, unsigned long long int w)
{
  ulonglong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) double1 make_double1(double x)
{
  double1 t; t.x = x; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) double2 make_double2(double x, double y)
{
  double2 t; t.x = x; t.y = y; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) double3 make_double3(double x, double y, double z)
{
  double3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __attribute__((host)) __attribute__((device)) double4 make_double4(double x, double y, double z, double w)
{
  double4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}
# 173 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/vector_functions.h" 2
# 102 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2
# 115 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 1
# 71 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 1
# 72 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 73 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 2
# 85 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
# 1 "/usr/include/string.h" 1 3 4
# 26 "/usr/include/string.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 27 "/usr/include/string.h" 2 3 4


# 28 "/usr/include/string.h" 3 4
extern "C" {




# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 1 3 4
# 34 "/usr/include/string.h" 2 3 4
# 43 "/usr/include/string.h" 3 4
extern void *memcpy (void *__restrict __dest, const void *__restrict __src,
       size_t __n) throw () __attribute__ ((__nonnull__ (1, 2)));


extern void *memmove (void *__dest, const void *__src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));





extern void *memccpy (void *__restrict __dest, const void *__restrict __src,
        int __c, size_t __n)
    throw () __attribute__ ((__nonnull__ (1, 2))) __attribute__ ((__access__ (__write_only__, 1, 4)));




extern void *memset (void *__s, int __c, size_t __n) throw () __attribute__ ((__nonnull__ (1)));


extern int memcmp (const void *__s1, const void *__s2, size_t __n)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));



extern "C++"
{
extern void *memchr (void *__s, int __c, size_t __n)
      throw () __asm ("memchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern const void *memchr (const void *__s, int __c, size_t __n)
      throw () __asm ("memchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 89 "/usr/include/string.h" 3 4
}
# 99 "/usr/include/string.h" 3 4
extern "C++" void *rawmemchr (void *__s, int __c)
     throw () __asm ("rawmemchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern "C++" const void *rawmemchr (const void *__s, int __c)
     throw () __asm ("rawmemchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));







extern "C++" void *memrchr (void *__s, int __c, size_t __n)
      throw () __asm ("memrchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)))
      __attribute__ ((__access__ (__read_only__, 1, 3)));
extern "C++" const void *memrchr (const void *__s, int __c, size_t __n)
      throw () __asm ("memrchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)))
      __attribute__ ((__access__ (__read_only__, 1, 3)));
# 125 "/usr/include/string.h" 3 4
extern char *strcpy (char *__restrict __dest, const char *__restrict __src)
     throw () __attribute__ ((__nonnull__ (1, 2)));

extern char *strncpy (char *__restrict __dest,
        const char *__restrict __src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));


extern char *strcat (char *__restrict __dest, const char *__restrict __src)
     throw () __attribute__ ((__nonnull__ (1, 2)));

extern char *strncat (char *__restrict __dest, const char *__restrict __src,
        size_t __n) throw () __attribute__ ((__nonnull__ (1, 2)));


extern int strcmp (const char *__s1, const char *__s2)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern int strncmp (const char *__s1, const char *__s2, size_t __n)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strcoll (const char *__s1, const char *__s2)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern size_t strxfrm (char *__restrict __dest,
         const char *__restrict __src, size_t __n)
    throw () __attribute__ ((__nonnull__ (2))) __attribute__ ((__access__ (__write_only__, 1, 3)));



# 1 "/usr/include/bits/types/locale_t.h" 1 3 4
# 22 "/usr/include/bits/types/locale_t.h" 3 4
# 1 "/usr/include/bits/types/__locale_t.h" 1 3 4
# 28 "/usr/include/bits/types/__locale_t.h" 3 4
struct __locale_struct
{

  struct __locale_data *__locales[13];


  const unsigned short int *__ctype_b;
  const int *__ctype_tolower;
  const int *__ctype_toupper;


  const char *__names[13];
};

typedef struct __locale_struct *__locale_t;
# 23 "/usr/include/bits/types/locale_t.h" 2 3 4

typedef __locale_t locale_t;
# 157 "/usr/include/string.h" 2 3 4


extern int strcoll_l (const char *__s1, const char *__s2, locale_t __l)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2, 3)));


extern size_t strxfrm_l (char *__dest, const char *__src, size_t __n,
    locale_t __l) throw () __attribute__ ((__nonnull__ (2, 4)))
     __attribute__ ((__access__ (__write_only__, 1, 3)));





extern char *strdup (const char *__s)
     throw () __attribute__ ((__malloc__)) __attribute__ ((__nonnull__ (1)));






extern char *strndup (const char *__string, size_t __n)
     throw () __attribute__ ((__malloc__)) __attribute__ ((__nonnull__ (1)));
# 208 "/usr/include/string.h" 3 4
extern "C++"
{
extern char *strchr (char *__s, int __c)
     throw () __asm ("strchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern const char *strchr (const char *__s, int __c)
     throw () __asm ("strchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 228 "/usr/include/string.h" 3 4
}






extern "C++"
{
extern char *strrchr (char *__s, int __c)
     throw () __asm ("strrchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern const char *strrchr (const char *__s, int __c)
     throw () __asm ("strrchr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 255 "/usr/include/string.h" 3 4
}
# 265 "/usr/include/string.h" 3 4
extern "C++" char *strchrnul (char *__s, int __c)
     throw () __asm ("strchrnul") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern "C++" const char *strchrnul (const char *__s, int __c)
     throw () __asm ("strchrnul") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 277 "/usr/include/string.h" 3 4
extern size_t strcspn (const char *__s, const char *__reject)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern size_t strspn (const char *__s, const char *__accept)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern "C++"
{
extern char *strpbrk (char *__s, const char *__accept)
     throw () __asm ("strpbrk") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
extern const char *strpbrk (const char *__s, const char *__accept)
     throw () __asm ("strpbrk") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
# 305 "/usr/include/string.h" 3 4
}






extern "C++"
{
extern char *strstr (char *__haystack, const char *__needle)
     throw () __asm ("strstr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
extern const char *strstr (const char *__haystack, const char *__needle)
     throw () __asm ("strstr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
# 332 "/usr/include/string.h" 3 4
}







extern char *strtok (char *__restrict __s, const char *__restrict __delim)
     throw () __attribute__ ((__nonnull__ (2)));



extern char *__strtok_r (char *__restrict __s,
    const char *__restrict __delim,
    char **__restrict __save_ptr)
     throw () __attribute__ ((__nonnull__ (2, 3)));

extern char *strtok_r (char *__restrict __s, const char *__restrict __delim,
         char **__restrict __save_ptr)
     throw () __attribute__ ((__nonnull__ (2, 3)));





extern "C++" char *strcasestr (char *__haystack, const char *__needle)
     throw () __asm ("strcasestr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
extern "C++" const char *strcasestr (const char *__haystack,
         const char *__needle)
     throw () __asm ("strcasestr") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
# 373 "/usr/include/string.h" 3 4
extern void *memmem (const void *__haystack, size_t __haystacklen,
       const void *__needle, size_t __needlelen)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 3)))
    __attribute__ ((__access__ (__read_only__, 1, 2)))
    __attribute__ ((__access__ (__read_only__, 3, 4)));



extern void *__mempcpy (void *__restrict __dest,
   const void *__restrict __src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));
extern void *mempcpy (void *__restrict __dest,
        const void *__restrict __src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));




extern size_t strlen (const char *__s)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));




extern size_t strnlen (const char *__string, size_t __maxlen)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));




extern char *strerror (int __errnum) throw ();
# 428 "/usr/include/string.h" 3 4
extern char *strerror_r (int __errnum, char *__buf, size_t __buflen)
     throw () __attribute__ ((__nonnull__ (2))) __attribute__ ((__access__ (__write_only__, 2, 3)));




extern const char *strerrordesc_np (int __err) throw ();

extern const char *strerrorname_np (int __err) throw ();





extern char *strerror_l (int __errnum, locale_t __l) throw ();



# 1 "/usr/include/strings.h" 1 3 4
# 23 "/usr/include/strings.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 1 3 4
# 24 "/usr/include/strings.h" 2 3 4






extern "C" {



extern int bcmp (const void *__s1, const void *__s2, size_t __n)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern void bcopy (const void *__src, void *__dest, size_t __n)
  throw () __attribute__ ((__nonnull__ (1, 2)));


extern void bzero (void *__s, size_t __n) throw () __attribute__ ((__nonnull__ (1)));



extern "C++"
{
extern char *index (char *__s, int __c)
     throw () __asm ("index") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern const char *index (const char *__s, int __c)
     throw () __asm ("index") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 66 "/usr/include/strings.h" 3 4
}







extern "C++"
{
extern char *rindex (char *__s, int __c)
     throw () __asm ("rindex") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
extern const char *rindex (const char *__s, int __c)
     throw () __asm ("rindex") __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 94 "/usr/include/strings.h" 3 4
}
# 104 "/usr/include/strings.h" 3 4
extern int ffs (int __i) throw () __attribute__ ((__const__));





extern int ffsl (long int __l) throw () __attribute__ ((__const__));
__extension__ extern int ffsll (long long int __ll)
     throw () __attribute__ ((__const__));



extern int strcasecmp (const char *__s1, const char *__s2)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strncasecmp (const char *__s1, const char *__s2, size_t __n)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));






extern int strcasecmp_l (const char *__s1, const char *__s2, locale_t __loc)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2, 3)));



extern int strncasecmp_l (const char *__s1, const char *__s2,
     size_t __n, locale_t __loc)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2, 4)));


}
# 447 "/usr/include/string.h" 2 3 4



extern void explicit_bzero (void *__s, size_t __n) throw () __attribute__ ((__nonnull__ (1)))
    __attribute__ ((__access__ (__write_only__, 1, 2)));



extern char *strsep (char **__restrict __stringp,
       const char *__restrict __delim)
     throw () __attribute__ ((__nonnull__ (1, 2)));




extern char *strsignal (int __sig) throw ();



extern const char *sigabbrev_np (int __sig) throw ();


extern const char *sigdescr_np (int __sig) throw ();



extern char *__stpcpy (char *__restrict __dest, const char *__restrict __src)
     throw () __attribute__ ((__nonnull__ (1, 2)));
extern char *stpcpy (char *__restrict __dest, const char *__restrict __src)
     throw () __attribute__ ((__nonnull__ (1, 2)));



extern char *__stpncpy (char *__restrict __dest,
   const char *__restrict __src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));
extern char *stpncpy (char *__restrict __dest,
        const char *__restrict __src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));




extern int strverscmp (const char *__s1, const char *__s2)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern char *strfry (char *__string) throw () __attribute__ ((__nonnull__ (1)));


extern void *memfrob (void *__s, size_t __n) throw () __attribute__ ((__nonnull__ (1)))
    __attribute__ ((__access__ (__write_only__, 1, 2)));







extern "C++" char *basename (char *__filename)
     throw () __asm ("basename") __attribute__ ((__nonnull__ (1)));
extern "C++" const char *basename (const char *__filename)
     throw () __asm ("basename") __attribute__ ((__nonnull__ (1)));
# 523 "/usr/include/string.h" 3 4
}
# 86 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 2
# 1 "/usr/include/time.h" 1 3 4
# 29 "/usr/include/time.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 1 3 4
# 30 "/usr/include/time.h" 2 3 4



# 1 "/usr/include/bits/time.h" 1 3 4
# 26 "/usr/include/bits/time.h" 3 4
# 1 "/usr/include/bits/types.h" 1 3 4
# 27 "/usr/include/bits/types.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 28 "/usr/include/bits/types.h" 2 3 4
# 1 "/usr/include/bits/timesize.h" 1 3 4
# 29 "/usr/include/bits/types.h" 2 3 4


typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;


typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;

typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;






typedef __int8_t __int_least8_t;
typedef __uint8_t __uint_least8_t;
typedef __int16_t __int_least16_t;
typedef __uint16_t __uint_least16_t;
typedef __int32_t __int_least32_t;
typedef __uint32_t __uint_least32_t;
typedef __int64_t __int_least64_t;
typedef __uint64_t __uint_least64_t;



typedef long int __quad_t;
typedef unsigned long int __u_quad_t;







typedef long int __intmax_t;
typedef unsigned long int __uintmax_t;
# 141 "/usr/include/bits/types.h" 3 4
# 1 "/usr/include/bits/typesizes.h" 1 3 4
# 142 "/usr/include/bits/types.h" 2 3 4
# 1 "/usr/include/bits/time64.h" 1 3 4
# 143 "/usr/include/bits/types.h" 2 3 4


typedef unsigned long int __dev_t;
typedef unsigned int __uid_t;
typedef unsigned int __gid_t;
typedef unsigned long int __ino_t;
typedef unsigned long int __ino64_t;
typedef unsigned int __mode_t;
typedef unsigned long int __nlink_t;
typedef long int __off_t;
typedef long int __off64_t;
typedef int __pid_t;
typedef struct { int __val[2]; } __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;
typedef long int __suseconds64_t;

typedef int __daddr_t;
typedef int __key_t;


typedef int __clockid_t;


typedef void * __timer_t;


typedef long int __blksize_t;




typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;


typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;


typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;


typedef long int __fsword_t;

typedef long int __ssize_t;


typedef long int __syscall_slong_t;

typedef unsigned long int __syscall_ulong_t;



typedef __off64_t __loff_t;
typedef char *__caddr_t;


typedef long int __intptr_t;


typedef unsigned int __socklen_t;




typedef int __sig_atomic_t;
# 27 "/usr/include/bits/time.h" 2 3 4
# 73 "/usr/include/bits/time.h" 3 4
# 1 "/usr/include/bits/timex.h" 1 3 4
# 22 "/usr/include/bits/timex.h" 3 4
# 1 "/usr/include/bits/types/struct_timeval.h" 1 3 4







struct timeval
{
  __time_t tv_sec;
  __suseconds_t tv_usec;
};
# 23 "/usr/include/bits/timex.h" 2 3 4



struct timex
{
  unsigned int modes;
  __syscall_slong_t offset;
  __syscall_slong_t freq;
  __syscall_slong_t maxerror;
  __syscall_slong_t esterror;
  int status;
  __syscall_slong_t constant;
  __syscall_slong_t precision;
  __syscall_slong_t tolerance;
  struct timeval time;
  __syscall_slong_t tick;
  __syscall_slong_t ppsfreq;
  __syscall_slong_t jitter;
  int shift;
  __syscall_slong_t stabil;
  __syscall_slong_t jitcnt;
  __syscall_slong_t calcnt;
  __syscall_slong_t errcnt;
  __syscall_slong_t stbcnt;

  int tai;


  int :32; int :32; int :32; int :32;
  int :32; int :32; int :32; int :32;
  int :32; int :32; int :32;
};
# 74 "/usr/include/bits/time.h" 2 3 4

extern "C" {


extern int clock_adjtime (__clockid_t __clock_id, struct timex *__utx) throw ();

}
# 34 "/usr/include/time.h" 2 3 4



# 1 "/usr/include/bits/types/clock_t.h" 1 3 4






typedef __clock_t clock_t;
# 38 "/usr/include/time.h" 2 3 4
# 1 "/usr/include/bits/types/time_t.h" 1 3 4






typedef __time_t time_t;
# 39 "/usr/include/time.h" 2 3 4
# 1 "/usr/include/bits/types/struct_tm.h" 1 3 4






struct tm
{
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;


  long int tm_gmtoff;
  const char *tm_zone;




};
# 40 "/usr/include/time.h" 2 3 4


# 1 "/usr/include/bits/types/struct_timespec.h" 1 3 4





# 1 "/usr/include/bits/endian.h" 1 3 4
# 35 "/usr/include/bits/endian.h" 3 4
# 1 "/usr/include/bits/endianness.h" 1 3 4
# 36 "/usr/include/bits/endian.h" 2 3 4
# 7 "/usr/include/bits/types/struct_timespec.h" 2 3 4



struct timespec
{
  __time_t tv_sec;



  __syscall_slong_t tv_nsec;
# 26 "/usr/include/bits/types/struct_timespec.h" 3 4
};
# 43 "/usr/include/time.h" 2 3 4



# 1 "/usr/include/bits/types/clockid_t.h" 1 3 4






typedef __clockid_t clockid_t;
# 47 "/usr/include/time.h" 2 3 4
# 1 "/usr/include/bits/types/timer_t.h" 1 3 4






typedef __timer_t timer_t;
# 48 "/usr/include/time.h" 2 3 4
# 1 "/usr/include/bits/types/struct_itimerspec.h" 1 3 4







struct itimerspec
  {
    struct timespec it_interval;
    struct timespec it_value;
  };
# 49 "/usr/include/time.h" 2 3 4
struct sigevent;




typedef __pid_t pid_t;
# 68 "/usr/include/time.h" 3 4
extern "C" {



extern clock_t clock (void) throw ();


extern time_t time (time_t *__timer) throw ();


extern double difftime (time_t __time1, time_t __time0)
     throw () __attribute__ ((__const__));


extern time_t mktime (struct tm *__tp) throw ();





extern size_t strftime (char *__restrict __s, size_t __maxsize,
   const char *__restrict __format,
   const struct tm *__restrict __tp) throw ();




extern char *strptime (const char *__restrict __s,
         const char *__restrict __fmt, struct tm *__tp)
     throw ();






extern size_t strftime_l (char *__restrict __s, size_t __maxsize,
     const char *__restrict __format,
     const struct tm *__restrict __tp,
     locale_t __loc) throw ();



extern char *strptime_l (const char *__restrict __s,
    const char *__restrict __fmt, struct tm *__tp,
    locale_t __loc) throw ();





extern struct tm *gmtime (const time_t *__timer) throw ();



extern struct tm *localtime (const time_t *__timer) throw ();




extern struct tm *gmtime_r (const time_t *__restrict __timer,
       struct tm *__restrict __tp) throw ();



extern struct tm *localtime_r (const time_t *__restrict __timer,
          struct tm *__restrict __tp) throw ();




extern char *asctime (const struct tm *__tp) throw ();


extern char *ctime (const time_t *__timer) throw ();






extern char *asctime_r (const struct tm *__restrict __tp,
   char *__restrict __buf) throw ();


extern char *ctime_r (const time_t *__restrict __timer,
        char *__restrict __buf) throw ();




extern char *__tzname[2];
extern int __daylight;
extern long int __timezone;




extern char *tzname[2];



extern void tzset (void) throw ();



extern int daylight;
extern long int timezone;
# 190 "/usr/include/time.h" 3 4
extern time_t timegm (struct tm *__tp) throw ();


extern time_t timelocal (struct tm *__tp) throw ();


extern int dysize (int __year) throw () __attribute__ ((__const__));
# 205 "/usr/include/time.h" 3 4
extern int nanosleep (const struct timespec *__requested_time,
        struct timespec *__remaining);



extern int clock_getres (clockid_t __clock_id, struct timespec *__res) throw ();


extern int clock_gettime (clockid_t __clock_id, struct timespec *__tp) throw ();


extern int clock_settime (clockid_t __clock_id, const struct timespec *__tp)
     throw ();






extern int clock_nanosleep (clockid_t __clock_id, int __flags,
       const struct timespec *__req,
       struct timespec *__rem);


extern int clock_getcpuclockid (pid_t __pid, clockid_t *__clock_id) throw ();




extern int timer_create (clockid_t __clock_id,
    struct sigevent *__restrict __evp,
    timer_t *__restrict __timerid) throw ();


extern int timer_delete (timer_t __timerid) throw ();


extern int timer_settime (timer_t __timerid, int __flags,
     const struct itimerspec *__restrict __value,
     struct itimerspec *__restrict __ovalue) throw ();


extern int timer_gettime (timer_t __timerid, struct itimerspec *__value)
     throw ();


extern int timer_getoverrun (timer_t __timerid) throw ();





extern int timespec_get (struct timespec *__ts, int __base)
     throw () __attribute__ ((__nonnull__ (1)));
# 274 "/usr/include/time.h" 3 4
extern int getdate_err;
# 283 "/usr/include/time.h" 3 4
extern struct tm *getdate (const char *__string);
# 297 "/usr/include/time.h" 3 4
extern int getdate_r (const char *__restrict __string,
        struct tm *__restrict __resbufp);


}
# 87 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 2


# 88 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern "C"
{

extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) clock_t clock(void)




# 95 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 3 4
throw ()
# 95 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
      ;
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) void* memset(void*, int, size_t) 
# 96 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 3 4
                                                                                                                   throw ()
# 96 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
                                                                                                                          ;
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) void* memcpy(void*, const void*, size_t) 
# 97 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 3 4
                                                                                                                           throw ()
# 97 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
                                                                                                                                  ;

}
# 111 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
# 1 "/usr/include/c++/10/new" 1 3
# 38 "/usr/include/c++/10/new" 3
       
# 39 "/usr/include/c++/10/new" 3

# 1 "/usr/include/c++/10/x86_64-redhat-linux/bits/c++config.h" 1 3


# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 4 "/usr/include/c++/10/x86_64-redhat-linux/bits/c++config.h" 2 3
# 2361 "/usr/include/c++/10/x86_64-redhat-linux/bits/c++config.h" 3

# 2361 "/usr/include/c++/10/x86_64-redhat-linux/bits/c++config.h" 3
namespace std
{
  typedef long unsigned int size_t;
  typedef long int ptrdiff_t;


  typedef decltype(nullptr) nullptr_t;

}
# 2383 "/usr/include/c++/10/x86_64-redhat-linux/bits/c++config.h" 3
namespace std
{
  inline namespace __cxx11 __attribute__((__abi_tag__ ("cxx11"))) { }
}
namespace __gnu_cxx
{
  inline namespace __cxx11 __attribute__((__abi_tag__ ("cxx11"))) { }
}
# 2621 "/usr/include/c++/10/x86_64-redhat-linux/bits/c++config.h" 3
# 1 "/usr/include/c++/10/x86_64-redhat-linux/bits/os_defines.h" 1 3
# 2622 "/usr/include/c++/10/x86_64-redhat-linux/bits/c++config.h" 2 3


# 1 "/usr/include/c++/10/x86_64-redhat-linux/bits/cpu_defines.h" 1 3
# 2625 "/usr/include/c++/10/x86_64-redhat-linux/bits/c++config.h" 2 3
# 41 "/usr/include/c++/10/new" 2 3
# 1 "/usr/include/c++/10/exception" 1 3
# 33 "/usr/include/c++/10/exception" 3
       
# 34 "/usr/include/c++/10/exception" 3

#pragma GCC visibility push(default)


# 1 "/usr/include/c++/10/bits/exception.h" 1 3
# 34 "/usr/include/c++/10/bits/exception.h" 3
       
# 35 "/usr/include/c++/10/bits/exception.h" 3

#pragma GCC visibility push(default)



extern "C++" {

namespace std
{
# 60 "/usr/include/c++/10/bits/exception.h" 3
  class exception
  {
  public:
    exception() noexcept { }
    virtual ~exception() noexcept;

    exception(const exception&) = default;
    exception& operator=(const exception&) = default;
    exception(exception&&) = default;
    exception& operator=(exception&&) = default;




    virtual const char*
    what() const noexcept;
  };

}

}

#pragma GCC visibility pop
# 39 "/usr/include/c++/10/exception" 2 3

extern "C++" {

namespace std
{






  class bad_exception : public exception
  {
  public:
    bad_exception() noexcept { }



    virtual ~bad_exception() noexcept;


    virtual const char*
    what() const noexcept;
  };


  typedef void (*terminate_handler) ();


  typedef void (*unexpected_handler) ();


  terminate_handler set_terminate(terminate_handler) noexcept;



  terminate_handler get_terminate() noexcept;




  void terminate() noexcept __attribute__ ((__noreturn__));


  unexpected_handler set_unexpected(unexpected_handler) noexcept;



  unexpected_handler get_unexpected() noexcept;




  void unexpected() __attribute__ ((__noreturn__));
# 105 "/usr/include/c++/10/exception" 3
 
  bool uncaught_exception() noexcept __attribute__ ((__pure__));




  int uncaught_exceptions() noexcept __attribute__ ((__pure__));



}

namespace __gnu_cxx
{

# 137 "/usr/include/c++/10/exception" 3
  void __verbose_terminate_handler();


}

}

#pragma GCC visibility pop


# 1 "/usr/include/c++/10/bits/exception_ptr.h" 1 3
# 34 "/usr/include/c++/10/bits/exception_ptr.h" 3
#pragma GCC visibility push(default)


# 1 "/usr/include/c++/10/bits/exception_defines.h" 1 3
# 38 "/usr/include/c++/10/bits/exception_ptr.h" 2 3
# 1 "/usr/include/c++/10/bits/cxxabi_init_exception.h" 1 3
# 34 "/usr/include/c++/10/bits/cxxabi_init_exception.h" 3
       
# 35 "/usr/include/c++/10/bits/cxxabi_init_exception.h" 3

#pragma GCC visibility push(default)

# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 1 3 4
# 39 "/usr/include/c++/10/bits/cxxabi_init_exception.h" 2 3
# 50 "/usr/include/c++/10/bits/cxxabi_init_exception.h" 3
namespace std
{
  class type_info;
}

namespace __cxxabiv1
{
  struct __cxa_refcounted_exception;

  extern "C"
    {

      void*
      __cxa_allocate_exception(size_t) noexcept;

      void
      __cxa_free_exception(void*) noexcept;


      __cxa_refcounted_exception*
      __cxa_init_primary_exception(void *object, std::type_info *tinfo,
                void ( *dest) (void *)) noexcept;

    }
}



#pragma GCC visibility pop
# 39 "/usr/include/c++/10/bits/exception_ptr.h" 2 3
# 1 "/usr/include/c++/10/typeinfo" 1 3
# 32 "/usr/include/c++/10/typeinfo" 3
       
# 33 "/usr/include/c++/10/typeinfo" 3



# 1 "/usr/include/c++/10/bits/hash_bytes.h" 1 3
# 33 "/usr/include/c++/10/bits/hash_bytes.h" 3
       
# 34 "/usr/include/c++/10/bits/hash_bytes.h" 3



namespace std
{







  size_t
  _Hash_bytes(const void* __ptr, size_t __len, size_t __seed);





  size_t
  _Fnv_hash_bytes(const void* __ptr, size_t __len, size_t __seed);


}
# 37 "/usr/include/c++/10/typeinfo" 2 3


#pragma GCC visibility push(default)

extern "C++" {

namespace __cxxabiv1
{
  class __class_type_info;
}
# 80 "/usr/include/c++/10/typeinfo" 3
namespace std
{






  class type_info
  {
  public:




    virtual ~type_info();



    const char* name() const noexcept
    { return __name[0] == '*' ? __name + 1 : __name; }
# 115 "/usr/include/c++/10/typeinfo" 3
    bool before(const type_info& __arg) const noexcept
    { return (__name[0] == '*' && __arg.__name[0] == '*')
 ? __name < __arg.__name
 : __builtin_strcmp (__name, __arg.__name) < 0; }

    bool operator==(const type_info& __arg) const noexcept
    {
      return ((__name == __arg.__name)
       || (__name[0] != '*' &&
    __builtin_strcmp (__name, __arg.__name) == 0));
    }
# 138 "/usr/include/c++/10/typeinfo" 3
    bool operator!=(const type_info& __arg) const noexcept
    { return !operator==(__arg); }



    size_t hash_code() const noexcept
    {

      return _Hash_bytes(name(), __builtin_strlen(name()),
    static_cast<size_t>(0xc70f6907UL));



    }



    virtual bool __is_pointer_p() const;


    virtual bool __is_function_p() const;







    virtual bool __do_catch(const type_info *__thr_type, void **__thr_obj,
       unsigned __outer) const;


    virtual bool __do_upcast(const __cxxabiv1::__class_type_info *__target,
        void **__obj_ptr) const;

  protected:
    const char *__name;

    explicit type_info(const char *__n): __name(__n) { }

  private:

    type_info& operator=(const type_info&);
    type_info(const type_info&);
  };







  class bad_cast : public exception
  {
  public:
    bad_cast() noexcept { }



    virtual ~bad_cast() noexcept;


    virtual const char* what() const noexcept;
  };





  class bad_typeid : public exception
  {
  public:
    bad_typeid () noexcept { }



    virtual ~bad_typeid() noexcept;


    virtual const char* what() const noexcept;
  };
}

}

#pragma GCC visibility pop
# 40 "/usr/include/c++/10/bits/exception_ptr.h" 2 3
# 1 "/usr/include/c++/10/new" 1 3
# 41 "/usr/include/c++/10/bits/exception_ptr.h" 2 3

extern "C++" {

namespace std
{
  class type_info;






  namespace __exception_ptr
  {
    class exception_ptr;
  }

  using __exception_ptr::exception_ptr;





  exception_ptr current_exception() noexcept;

  template<typename _Ex>
  exception_ptr make_exception_ptr(_Ex) noexcept;


  void rethrow_exception(exception_ptr) __attribute__ ((__noreturn__));

  namespace __exception_ptr
  {
    using std::rethrow_exception;





    class exception_ptr
    {
      void* _M_exception_object;

      explicit exception_ptr(void* __e) noexcept;

      void _M_addref() noexcept;
      void _M_release() noexcept;

      void *_M_get() const noexcept __attribute__ ((__pure__));

      friend exception_ptr std::current_exception() noexcept;
      friend void std::rethrow_exception(exception_ptr);
      template<typename _Ex>
      friend exception_ptr std::make_exception_ptr(_Ex) noexcept;

    public:
      exception_ptr() noexcept;

      exception_ptr(const exception_ptr&) noexcept;


      exception_ptr(nullptr_t) noexcept
      : _M_exception_object(0)
      { }

      exception_ptr(exception_ptr&& __o) noexcept
      : _M_exception_object(__o._M_exception_object)
      { __o._M_exception_object = 0; }
# 118 "/usr/include/c++/10/bits/exception_ptr.h" 3
      exception_ptr&
      operator=(const exception_ptr&) noexcept;


      exception_ptr&
      operator=(exception_ptr&& __o) noexcept
      {
        exception_ptr(static_cast<exception_ptr&&>(__o)).swap(*this);
        return *this;
      }


      ~exception_ptr() noexcept;

      void
      swap(exception_ptr&) noexcept;
# 145 "/usr/include/c++/10/bits/exception_ptr.h" 3
      explicit operator bool() const
      { return _M_exception_object; }


      friend bool
      operator==(const exception_ptr&, const exception_ptr&)
 noexcept __attribute__ ((__pure__));

      const class std::type_info*
      __cxa_exception_type() const noexcept
 __attribute__ ((__pure__));
    };



    bool
    operator==(const exception_ptr&, const exception_ptr&)
      noexcept __attribute__ ((__pure__));

    bool
    operator!=(const exception_ptr&, const exception_ptr&)
      noexcept __attribute__ ((__pure__));

    inline void
    swap(exception_ptr& __lhs, exception_ptr& __rhs)
    { __lhs.swap(__rhs); }




    template<typename _Ex>
      inline void
      __dest_thunk(void* __x)
      { static_cast<_Ex*>(__x)->~_Ex(); }


  }


  template<typename _Ex>
    exception_ptr
    make_exception_ptr(_Ex __ex) noexcept
    {

      void* __e = __cxxabiv1::__cxa_allocate_exception(sizeof(_Ex));
      (void) __cxxabiv1::__cxa_init_primary_exception(
   __e, const_cast<std::type_info*>(&typeid(__ex)),
   __exception_ptr::__dest_thunk<_Ex>);
      try
 {
          ::new (__e) _Ex(__ex);
          return exception_ptr(__e);
 }
      catch(...)
 {
   __cxxabiv1::__cxa_free_exception(__e);
   return current_exception();
 }
# 215 "/usr/include/c++/10/bits/exception_ptr.h" 3
    }


}

}

#pragma GCC visibility pop
# 148 "/usr/include/c++/10/exception" 2 3
# 1 "/usr/include/c++/10/bits/nested_exception.h" 1 3
# 33 "/usr/include/c++/10/bits/nested_exception.h" 3
#pragma GCC visibility push(default)






# 1 "/usr/include/c++/10/bits/move.h" 1 3
# 38 "/usr/include/c++/10/bits/move.h" 3
namespace std __attribute__ ((__visibility__ ("default")))
{







  template<typename _Tp>
    inline constexpr _Tp*
    __addressof(_Tp& __r) noexcept
    { return __builtin_addressof(__r); }




}

# 1 "/usr/include/c++/10/type_traits" 1 3
# 32 "/usr/include/c++/10/type_traits" 3
       
# 33 "/usr/include/c++/10/type_traits" 3







namespace std __attribute__ ((__visibility__ ("default")))
{

# 56 "/usr/include/c++/10/type_traits" 3
  template<typename _Tp, _Tp __v>
    struct integral_constant
    {
      static constexpr _Tp value = __v;
      typedef _Tp value_type;
      typedef integral_constant<_Tp, __v> type;
      constexpr operator value_type() const noexcept { return value; }




      constexpr value_type operator()() const noexcept { return value; }

    };

  template<typename _Tp, _Tp __v>
    constexpr _Tp integral_constant<_Tp, __v>::value;


  typedef integral_constant<bool, true> true_type;


  typedef integral_constant<bool, false> false_type;

  template<bool __v>
    using __bool_constant = integral_constant<bool, __v>;
# 91 "/usr/include/c++/10/type_traits" 3
  template<bool, typename, typename>
    struct conditional;

  template <typename _Type>
    struct __type_identity
    { using type = _Type; };

  template<typename _Tp>
    using __type_identity_t = typename __type_identity<_Tp>::type;

  template<typename...>
    struct __or_;

  template<>
    struct __or_<>
    : public false_type
    { };

  template<typename _B1>
    struct __or_<_B1>
    : public _B1
    { };

  template<typename _B1, typename _B2>
    struct __or_<_B1, _B2>
    : public conditional<_B1::value, _B1, _B2>::type
    { };

  template<typename _B1, typename _B2, typename _B3, typename... _Bn>
    struct __or_<_B1, _B2, _B3, _Bn...>
    : public conditional<_B1::value, _B1, __or_<_B2, _B3, _Bn...>>::type
    { };

  template<typename...>
    struct __and_;

  template<>
    struct __and_<>
    : public true_type
    { };

  template<typename _B1>
    struct __and_<_B1>
    : public _B1
    { };

  template<typename _B1, typename _B2>
    struct __and_<_B1, _B2>
    : public conditional<_B1::value, _B2, _B1>::type
    { };

  template<typename _B1, typename _B2, typename _B3, typename... _Bn>
    struct __and_<_B1, _B2, _B3, _Bn...>
    : public conditional<_B1::value, __and_<_B2, _B3, _Bn...>, _B1>::type
    { };

  template<typename _Pp>
    struct __not_
    : public __bool_constant<!bool(_Pp::value)>
    { };
# 188 "/usr/include/c++/10/type_traits" 3
  template<typename>
    struct is_reference;
  template<typename>
    struct is_function;
  template<typename>
    struct is_void;
  template<typename>
    struct __is_array_unknown_bounds;




  template <typename _Tp, size_t = sizeof(_Tp)>
    constexpr true_type __is_complete_or_unbounded(__type_identity<_Tp>)
    { return {}; }

  template <typename _TypeIdentity,
      typename _NestedType = typename _TypeIdentity::type>
    constexpr typename __or_<
      is_reference<_NestedType>,
      is_function<_NestedType>,
      is_void<_NestedType>,
      __is_array_unknown_bounds<_NestedType>
    >::type __is_complete_or_unbounded(_TypeIdentity)
    { return {}; }






  template<typename _Tp>
    struct __success_type
    { typedef _Tp type; };

  struct __failure_type
  { };

  template<typename>
    struct remove_cv;


  template<typename _Tp>
    using __remove_cv_t = typename remove_cv<_Tp>::type;

  template<typename>
    struct is_const;



  template<typename>
    struct __is_void_helper
    : public false_type { };

  template<>
    struct __is_void_helper<void>
    : public true_type { };


  template<typename _Tp>
    struct is_void
    : public __is_void_helper<__remove_cv_t<_Tp>>::type
    { };

  template<typename>
    struct __is_integral_helper
    : public false_type { };

  template<>
    struct __is_integral_helper<bool>
    : public true_type { };

  template<>
    struct __is_integral_helper<char>
    : public true_type { };

  template<>
    struct __is_integral_helper<signed char>
    : public true_type { };

  template<>
    struct __is_integral_helper<unsigned char>
    : public true_type { };


  template<>
    struct __is_integral_helper<wchar_t>
    : public true_type { };
# 284 "/usr/include/c++/10/type_traits" 3
  template<>
    struct __is_integral_helper<char16_t>
    : public true_type { };

  template<>
    struct __is_integral_helper<char32_t>
    : public true_type { };

  template<>
    struct __is_integral_helper<short>
    : public true_type { };

  template<>
    struct __is_integral_helper<unsigned short>
    : public true_type { };

  template<>
    struct __is_integral_helper<int>
    : public true_type { };

  template<>
    struct __is_integral_helper<unsigned int>
    : public true_type { };

  template<>
    struct __is_integral_helper<long>
    : public true_type { };

  template<>
    struct __is_integral_helper<unsigned long>
    : public true_type { };

  template<>
    struct __is_integral_helper<long long>
    : public true_type { };

  template<>
    struct __is_integral_helper<unsigned long long>
    : public true_type { };




  template<>
    struct __is_integral_helper<__int128>
    : public true_type { };

  template<>
    struct __is_integral_helper<unsigned __int128>
    : public true_type { };
# 364 "/usr/include/c++/10/type_traits" 3
  template<typename _Tp>
    struct is_integral
    : public __is_integral_helper<__remove_cv_t<_Tp>>::type
    { };

  template<typename>
    struct __is_floating_point_helper
    : public false_type { };

  template<>
    struct __is_floating_point_helper<float>
    : public true_type { };

  template<>
    struct __is_floating_point_helper<double>
    : public true_type { };

  template<>
    struct __is_floating_point_helper<long double>
    : public true_type { };


  template<>
    struct __is_floating_point_helper<__float128>
    : public true_type { };



  template<typename _Tp>
    struct is_floating_point
    : public __is_floating_point_helper<__remove_cv_t<_Tp>>::type
    { };


  template<typename>
    struct is_array
    : public false_type { };

  template<typename _Tp, std::size_t _Size>
    struct is_array<_Tp[_Size]>
    : public true_type { };

  template<typename _Tp>
    struct is_array<_Tp[]>
    : public true_type { };

  template<typename>
    struct __is_pointer_helper
    : public false_type { };

  template<typename _Tp>
    struct __is_pointer_helper<_Tp*>
    : public true_type { };


  template<typename _Tp>
    struct is_pointer
    : public __is_pointer_helper<__remove_cv_t<_Tp>>::type
    { };


  template<typename>
    struct is_lvalue_reference
    : public false_type { };

  template<typename _Tp>
    struct is_lvalue_reference<_Tp&>
    : public true_type { };


  template<typename>
    struct is_rvalue_reference
    : public false_type { };

  template<typename _Tp>
    struct is_rvalue_reference<_Tp&&>
    : public true_type { };

  template<typename>
    struct __is_member_object_pointer_helper
    : public false_type { };

  template<typename _Tp, typename _Cp>
    struct __is_member_object_pointer_helper<_Tp _Cp::*>
    : public __not_<is_function<_Tp>>::type { };


  template<typename _Tp>
    struct is_member_object_pointer
    : public __is_member_object_pointer_helper<__remove_cv_t<_Tp>>::type
    { };

  template<typename>
    struct __is_member_function_pointer_helper
    : public false_type { };

  template<typename _Tp, typename _Cp>
    struct __is_member_function_pointer_helper<_Tp _Cp::*>
    : public is_function<_Tp>::type { };


  template<typename _Tp>
    struct is_member_function_pointer
    : public __is_member_function_pointer_helper<__remove_cv_t<_Tp>>::type
    { };


  template<typename _Tp>
    struct is_enum
    : public integral_constant<bool, __is_enum(_Tp)>
    { };


  template<typename _Tp>
    struct is_union
    : public integral_constant<bool, __is_union(_Tp)>
    { };


  template<typename _Tp>
    struct is_class
    : public integral_constant<bool, __is_class(_Tp)>
    { };


  template<typename _Tp>
    struct is_function
    : public __bool_constant<!is_const<const _Tp>::value> { };

  template<typename _Tp>
    struct is_function<_Tp&>
    : public false_type { };

  template<typename _Tp>
    struct is_function<_Tp&&>
    : public false_type { };



  template<typename>
    struct __is_null_pointer_helper
    : public false_type { };

  template<>
    struct __is_null_pointer_helper<std::nullptr_t>
    : public true_type { };


  template<typename _Tp>
    struct is_null_pointer
    : public __is_null_pointer_helper<__remove_cv_t<_Tp>>::type
    { };


  template<typename _Tp>
    struct __is_nullptr_t
    : public is_null_pointer<_Tp>
    { } __attribute__ ((__deprecated__ ("use '" "std::is_null_pointer" "' instead")));




  template<typename _Tp>
    struct is_reference
    : public __or_<is_lvalue_reference<_Tp>,
                   is_rvalue_reference<_Tp>>::type
    { };


  template<typename _Tp>
    struct is_arithmetic
    : public __or_<is_integral<_Tp>, is_floating_point<_Tp>>::type
    { };


  template<typename _Tp>
    struct is_fundamental
    : public __or_<is_arithmetic<_Tp>, is_void<_Tp>,
     is_null_pointer<_Tp>>::type
    { };


  template<typename _Tp>
    struct is_object
    : public __not_<__or_<is_function<_Tp>, is_reference<_Tp>,
                          is_void<_Tp>>>::type
    { };

  template<typename>
    struct is_member_pointer;


  template<typename _Tp>
    struct is_scalar
    : public __or_<is_arithmetic<_Tp>, is_enum<_Tp>, is_pointer<_Tp>,
                   is_member_pointer<_Tp>, is_null_pointer<_Tp>>::type
    { };


  template<typename _Tp>
    struct is_compound
    : public __not_<is_fundamental<_Tp>>::type { };

  template<typename _Tp>
    struct __is_member_pointer_helper
    : public false_type { };

  template<typename _Tp, typename _Cp>
    struct __is_member_pointer_helper<_Tp _Cp::*>
    : public true_type { };


  template<typename _Tp>
    struct is_member_pointer
    : public __is_member_pointer_helper<__remove_cv_t<_Tp>>::type
    { };

  template<typename, typename>
    struct is_same;

  template<typename _Tp, typename... _Types>
    using __is_one_of = __or_<is_same<_Tp, _Types>...>;


  template<typename _Tp>
    using __is_signed_integer = __is_one_of<__remove_cv_t<_Tp>,
   signed char, signed short, signed int, signed long,
   signed long long

   , signed __int128
# 604 "/usr/include/c++/10/type_traits" 3
   >;


  template<typename _Tp>
    using __is_unsigned_integer = __is_one_of<__remove_cv_t<_Tp>,
   unsigned char, unsigned short, unsigned int, unsigned long,
   unsigned long long

   , unsigned __int128
# 623 "/usr/include/c++/10/type_traits" 3
   >;


  template<typename _Tp>
    using __is_standard_integer
      = __or_<__is_signed_integer<_Tp>, __is_unsigned_integer<_Tp>>;


  template<typename...> using __void_t = void;



  template<typename _Tp, typename = void>
    struct __is_referenceable
    : public false_type
    { };

  template<typename _Tp>
    struct __is_referenceable<_Tp, __void_t<_Tp&>>
    : public true_type
    { };




  template<typename>
    struct is_const
    : public false_type { };

  template<typename _Tp>
    struct is_const<_Tp const>
    : public true_type { };


  template<typename>
    struct is_volatile
    : public false_type { };

  template<typename _Tp>
    struct is_volatile<_Tp volatile>
    : public true_type { };


  template<typename _Tp>
    struct is_trivial
    : public integral_constant<bool, __is_trivial(_Tp)>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp>
    struct is_trivially_copyable
    : public integral_constant<bool, __is_trivially_copyable(_Tp)>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp>
    struct is_standard_layout
    : public integral_constant<bool, __is_standard_layout(_Tp)>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };



  template<typename _Tp>
    struct
   
    is_pod
    : public integral_constant<bool, __is_pod(_Tp)>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp>
    struct is_literal_type
    : public integral_constant<bool, __is_literal_type(_Tp)>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp>
    struct is_empty
    : public integral_constant<bool, __is_empty(_Tp)>
    { };


  template<typename _Tp>
    struct is_polymorphic
    : public integral_constant<bool, __is_polymorphic(_Tp)>
    { };




  template<typename _Tp>
    struct is_final
    : public integral_constant<bool, __is_final(_Tp)>
    { };



  template<typename _Tp>
    struct is_abstract
    : public integral_constant<bool, __is_abstract(_Tp)>
    { };

  template<typename _Tp,
    bool = is_arithmetic<_Tp>::value>
    struct __is_signed_helper
    : public false_type { };

  template<typename _Tp>
    struct __is_signed_helper<_Tp, true>
    : public integral_constant<bool, _Tp(-1) < _Tp(0)>
    { };


  template<typename _Tp>
    struct is_signed
    : public __is_signed_helper<_Tp>::type
    { };


  template<typename _Tp>
    struct is_unsigned
    : public __and_<is_arithmetic<_Tp>, __not_<is_signed<_Tp>>>
    { };
# 770 "/usr/include/c++/10/type_traits" 3
  template<typename _Tp, typename _Up = _Tp&&>
    _Up
    __declval(int);

  template<typename _Tp>
    _Tp
    __declval(long);

  template<typename _Tp>
    auto declval() noexcept -> decltype(__declval<_Tp>(0));

  template<typename, unsigned = 0>
    struct extent;

  template<typename>
    struct remove_all_extents;

  template<typename _Tp>
    struct __is_array_known_bounds
    : public integral_constant<bool, (extent<_Tp>::value > 0)>
    { };

  template<typename _Tp>
    struct __is_array_unknown_bounds
    : public __and_<is_array<_Tp>, __not_<extent<_Tp>>>
    { };






  struct __do_is_destructible_impl
  {
    template<typename _Tp, typename = decltype(declval<_Tp&>().~_Tp())>
      static true_type __test(int);

    template<typename>
      static false_type __test(...);
  };

  template<typename _Tp>
    struct __is_destructible_impl
    : public __do_is_destructible_impl
    {
      typedef decltype(__test<_Tp>(0)) type;
    };

  template<typename _Tp,
           bool = __or_<is_void<_Tp>,
                        __is_array_unknown_bounds<_Tp>,
                        is_function<_Tp>>::value,
           bool = __or_<is_reference<_Tp>, is_scalar<_Tp>>::value>
    struct __is_destructible_safe;

  template<typename _Tp>
    struct __is_destructible_safe<_Tp, false, false>
    : public __is_destructible_impl<typename
               remove_all_extents<_Tp>::type>::type
    { };

  template<typename _Tp>
    struct __is_destructible_safe<_Tp, true, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_destructible_safe<_Tp, false, true>
    : public true_type { };


  template<typename _Tp>
    struct is_destructible
    : public __is_destructible_safe<_Tp>::type
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };





  struct __do_is_nt_destructible_impl
  {
    template<typename _Tp>
      static __bool_constant<noexcept(declval<_Tp&>().~_Tp())>
      __test(int);

    template<typename>
      static false_type __test(...);
  };

  template<typename _Tp>
    struct __is_nt_destructible_impl
    : public __do_is_nt_destructible_impl
    {
      typedef decltype(__test<_Tp>(0)) type;
    };

  template<typename _Tp,
           bool = __or_<is_void<_Tp>,
                        __is_array_unknown_bounds<_Tp>,
                        is_function<_Tp>>::value,
           bool = __or_<is_reference<_Tp>, is_scalar<_Tp>>::value>
    struct __is_nt_destructible_safe;

  template<typename _Tp>
    struct __is_nt_destructible_safe<_Tp, false, false>
    : public __is_nt_destructible_impl<typename
               remove_all_extents<_Tp>::type>::type
    { };

  template<typename _Tp>
    struct __is_nt_destructible_safe<_Tp, true, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_nt_destructible_safe<_Tp, false, true>
    : public true_type { };


  template<typename _Tp>
    struct is_nothrow_destructible
    : public __is_nt_destructible_safe<_Tp>::type
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, typename... _Args>
    struct __is_constructible_impl
    : public __bool_constant<__is_constructible(_Tp, _Args...)>
    { };


  template<typename _Tp, typename... _Args>
    struct is_constructible
      : public __is_constructible_impl<_Tp, _Args...>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp>
    struct is_default_constructible
    : public __is_constructible_impl<_Tp>::type
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_copy_constructible_impl;

  template<typename _Tp>
    struct __is_copy_constructible_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_copy_constructible_impl<_Tp, true>
    : public __is_constructible_impl<_Tp, const _Tp&>
    { };


  template<typename _Tp>
    struct is_copy_constructible
    : public __is_copy_constructible_impl<_Tp>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_move_constructible_impl;

  template<typename _Tp>
    struct __is_move_constructible_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_move_constructible_impl<_Tp, true>
    : public __is_constructible_impl<_Tp, _Tp&&>
    { };


  template<typename _Tp>
    struct is_move_constructible
    : public __is_move_constructible_impl<_Tp>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<bool, typename _Tp, typename... _Args>
    struct __is_nt_constructible_impl
    : public false_type
    { };

  template<typename _Tp, typename... _Args>
    struct __is_nt_constructible_impl<true, _Tp, _Args...>
    : public __bool_constant<noexcept(_Tp(std::declval<_Args>()...))>
    { };

  template<typename _Tp, typename _Arg>
    struct __is_nt_constructible_impl<true, _Tp, _Arg>
    : public __bool_constant<noexcept(static_cast<_Tp>(std::declval<_Arg>()))>
    { };

  template<typename _Tp>
    struct __is_nt_constructible_impl<true, _Tp>
    : public __bool_constant<noexcept(_Tp())>
    { };

  template<typename _Tp, size_t _Num>
    struct __is_nt_constructible_impl<true, _Tp[_Num]>
    : public __bool_constant<noexcept(typename remove_all_extents<_Tp>::type())>
    { };
# 1001 "/usr/include/c++/10/type_traits" 3
  template<typename _Tp, typename... _Args>
    using __is_nothrow_constructible_impl
      = __is_nt_constructible_impl<__is_constructible(_Tp, _Args...),
       _Tp, _Args...>;


  template<typename _Tp, typename... _Args>
    struct is_nothrow_constructible
    : public __is_nothrow_constructible_impl<_Tp, _Args...>::type
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp>
    struct is_nothrow_default_constructible
    : public __is_nothrow_constructible_impl<_Tp>::type
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_nothrow_copy_constructible_impl;

  template<typename _Tp>
    struct __is_nothrow_copy_constructible_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_nothrow_copy_constructible_impl<_Tp, true>
    : public __is_nothrow_constructible_impl<_Tp, const _Tp&>
    { };


  template<typename _Tp>
    struct is_nothrow_copy_constructible
    : public __is_nothrow_copy_constructible_impl<_Tp>::type
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_nothrow_move_constructible_impl;

  template<typename _Tp>
    struct __is_nothrow_move_constructible_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_nothrow_move_constructible_impl<_Tp, true>
    : public __is_nothrow_constructible_impl<_Tp, _Tp&&>
    { };


  template<typename _Tp>
    struct is_nothrow_move_constructible
    : public __is_nothrow_move_constructible_impl<_Tp>::type
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp, typename _Up>
    struct is_assignable
    : public __bool_constant<__is_assignable(_Tp, _Up)>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_copy_assignable_impl;

  template<typename _Tp>
    struct __is_copy_assignable_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_copy_assignable_impl<_Tp, true>
    : public __bool_constant<__is_assignable(_Tp&, const _Tp&)>
    { };


  template<typename _Tp>
    struct is_copy_assignable
    : public __is_copy_assignable_impl<_Tp>::type
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_move_assignable_impl;

  template<typename _Tp>
    struct __is_move_assignable_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_move_assignable_impl<_Tp, true>
    : public __bool_constant<__is_assignable(_Tp&, _Tp&&)>
    { };


  template<typename _Tp>
    struct is_move_assignable
    : public __is_move_assignable_impl<_Tp>::type
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, typename _Up>
    struct __is_nt_assignable_impl
    : public integral_constant<bool, noexcept(declval<_Tp>() = declval<_Up>())>
    { };

  template<typename _Tp, typename _Up>
    struct __is_nothrow_assignable_impl
    : public __and_<__bool_constant<__is_assignable(_Tp, _Up)>,
      __is_nt_assignable_impl<_Tp, _Up>>
    { };


  template<typename _Tp, typename _Up>
    struct is_nothrow_assignable
    : public __is_nothrow_assignable_impl<_Tp, _Up>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_nt_copy_assignable_impl;

  template<typename _Tp>
    struct __is_nt_copy_assignable_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_nt_copy_assignable_impl<_Tp, true>
    : public __is_nothrow_assignable_impl<_Tp&, const _Tp&>
    { };


  template<typename _Tp>
    struct is_nothrow_copy_assignable
    : public __is_nt_copy_assignable_impl<_Tp>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_nt_move_assignable_impl;

  template<typename _Tp>
    struct __is_nt_move_assignable_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_nt_move_assignable_impl<_Tp, true>
    : public __is_nothrow_assignable_impl<_Tp&, _Tp&&>
    { };


  template<typename _Tp>
    struct is_nothrow_move_assignable
    : public __is_nt_move_assignable_impl<_Tp>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp, typename... _Args>
    struct is_trivially_constructible
    : public __bool_constant<__is_trivially_constructible(_Tp, _Args...)>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp>
    struct is_trivially_default_constructible
    : public __bool_constant<__is_trivially_constructible(_Tp)>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  struct __do_is_implicitly_default_constructible_impl
  {
    template <typename _Tp>
    static void __helper(const _Tp&);

    template <typename _Tp>
    static true_type __test(const _Tp&,
                            decltype(__helper<const _Tp&>({}))* = 0);

    static false_type __test(...);
  };

  template<typename _Tp>
    struct __is_implicitly_default_constructible_impl
    : public __do_is_implicitly_default_constructible_impl
    {
      typedef decltype(__test(declval<_Tp>())) type;
    };

  template<typename _Tp>
    struct __is_implicitly_default_constructible_safe
    : public __is_implicitly_default_constructible_impl<_Tp>::type
    { };

  template <typename _Tp>
    struct __is_implicitly_default_constructible
    : public __and_<__is_constructible_impl<_Tp>,
      __is_implicitly_default_constructible_safe<_Tp>>
    { };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_trivially_copy_constructible_impl;

  template<typename _Tp>
    struct __is_trivially_copy_constructible_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_trivially_copy_constructible_impl<_Tp, true>
    : public __and_<__is_copy_constructible_impl<_Tp>,
      integral_constant<bool,
   __is_trivially_constructible(_Tp, const _Tp&)>>
    { };


  template<typename _Tp>
    struct is_trivially_copy_constructible
    : public __is_trivially_copy_constructible_impl<_Tp>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_trivially_move_constructible_impl;

  template<typename _Tp>
    struct __is_trivially_move_constructible_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_trivially_move_constructible_impl<_Tp, true>
    : public __and_<__is_move_constructible_impl<_Tp>,
      integral_constant<bool,
   __is_trivially_constructible(_Tp, _Tp&&)>>
    { };


  template<typename _Tp>
    struct is_trivially_move_constructible
    : public __is_trivially_move_constructible_impl<_Tp>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp, typename _Up>
    struct is_trivially_assignable
    : public __bool_constant<__is_trivially_assignable(_Tp, _Up)>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_trivially_copy_assignable_impl;

  template<typename _Tp>
    struct __is_trivially_copy_assignable_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_trivially_copy_assignable_impl<_Tp, true>
    : public __bool_constant<__is_trivially_assignable(_Tp&, const _Tp&)>
    { };


  template<typename _Tp>
    struct is_trivially_copy_assignable
    : public __is_trivially_copy_assignable_impl<_Tp>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __is_trivially_move_assignable_impl;

  template<typename _Tp>
    struct __is_trivially_move_assignable_impl<_Tp, false>
    : public false_type { };

  template<typename _Tp>
    struct __is_trivially_move_assignable_impl<_Tp, true>
    : public __bool_constant<__is_trivially_assignable(_Tp&, _Tp&&)>
    { };


  template<typename _Tp>
    struct is_trivially_move_assignable
    : public __is_trivially_move_assignable_impl<_Tp>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp>
    struct is_trivially_destructible
    : public __and_<__is_destructible_safe<_Tp>,
      __bool_constant<__has_trivial_destructor(_Tp)>>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };



  template<typename _Tp>
    struct has_virtual_destructor
    : public integral_constant<bool, __has_virtual_destructor(_Tp)>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };





  template<typename _Tp>
    struct alignment_of
    : public integral_constant<std::size_t, alignof(_Tp)>
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename>
    struct rank
    : public integral_constant<std::size_t, 0> { };

  template<typename _Tp, std::size_t _Size>
    struct rank<_Tp[_Size]>
    : public integral_constant<std::size_t, 1 + rank<_Tp>::value> { };

  template<typename _Tp>
    struct rank<_Tp[]>
    : public integral_constant<std::size_t, 1 + rank<_Tp>::value> { };


  template<typename, unsigned _Uint>
    struct extent
    : public integral_constant<std::size_t, 0> { };

  template<typename _Tp, unsigned _Uint, std::size_t _Size>
    struct extent<_Tp[_Size], _Uint>
    : public integral_constant<std::size_t,
          _Uint == 0 ? _Size : extent<_Tp,
          _Uint - 1>::value>
    { };

  template<typename _Tp, unsigned _Uint>
    struct extent<_Tp[], _Uint>
    : public integral_constant<std::size_t,
          _Uint == 0 ? 0 : extent<_Tp,
             _Uint - 1>::value>
    { };





  template<typename _Tp, typename _Up>
    struct is_same

    : public integral_constant<bool, __is_same_as(_Tp, _Up)>



    { };
# 1410 "/usr/include/c++/10/type_traits" 3
  template<typename _Base, typename _Derived>
    struct is_base_of
    : public integral_constant<bool, __is_base_of(_Base, _Derived)>
    { };

  template<typename _From, typename _To,
           bool = __or_<is_void<_From>, is_function<_To>,
                        is_array<_To>>::value>
    struct __is_convertible_helper
    {
      typedef typename is_void<_To>::type type;
    };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
  template<typename _From, typename _To>
    class __is_convertible_helper<_From, _To, false>
    {
      template<typename _To1>
 static void __test_aux(_To1) noexcept;

      template<typename _From1, typename _To1,
        typename = decltype(__test_aux<_To1>(std::declval<_From1>()))>
 static true_type
 __test(int);

      template<typename, typename>
 static false_type
 __test(...);

    public:
      typedef decltype(__test<_From, _To>(0)) type;
    };
#pragma GCC diagnostic pop


  template<typename _From, typename _To>
    struct is_convertible
    : public __is_convertible_helper<_From, _To>::type
    { };


  template<typename _ToElementType, typename _FromElementType>
    using __is_array_convertible
      = is_convertible<_FromElementType(*)[], _ToElementType(*)[]>;

  template<typename _From, typename _To,
           bool = __or_<is_void<_From>, is_function<_To>,
                        is_array<_To>>::value>
    struct __is_nt_convertible_helper
    : is_void<_To>
    { };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
  template<typename _From, typename _To>
    class __is_nt_convertible_helper<_From, _To, false>
    {
      template<typename _To1>
 static void __test_aux(_To1) noexcept;

      template<typename _From1, typename _To1>
 static
 __bool_constant<noexcept(__test_aux<_To1>(std::declval<_From1>()))>
 __test(int);

      template<typename, typename>
 static false_type
 __test(...);

    public:
      using type = decltype(__test<_From, _To>(0));
    };
#pragma GCC diagnostic pop


  template<typename _From, typename _To>
    struct __is_nothrow_convertible
    : public __is_nt_convertible_helper<_From, _To>::type
    { };
# 1508 "/usr/include/c++/10/type_traits" 3
  template<typename _Tp>
    struct remove_const
    { typedef _Tp type; };

  template<typename _Tp>
    struct remove_const<_Tp const>
    { typedef _Tp type; };


  template<typename _Tp>
    struct remove_volatile
    { typedef _Tp type; };

  template<typename _Tp>
    struct remove_volatile<_Tp volatile>
    { typedef _Tp type; };


  template<typename _Tp>
    struct remove_cv
    { using type = _Tp; };

  template<typename _Tp>
    struct remove_cv<const _Tp>
    { using type = _Tp; };

  template<typename _Tp>
    struct remove_cv<volatile _Tp>
    { using type = _Tp; };

  template<typename _Tp>
    struct remove_cv<const volatile _Tp>
    { using type = _Tp; };


  template<typename _Tp>
    struct add_const
    { typedef _Tp const type; };


  template<typename _Tp>
    struct add_volatile
    { typedef _Tp volatile type; };


  template<typename _Tp>
    struct add_cv
    {
      typedef typename
      add_const<typename add_volatile<_Tp>::type>::type type;
    };






  template<typename _Tp>
    using remove_const_t = typename remove_const<_Tp>::type;


  template<typename _Tp>
    using remove_volatile_t = typename remove_volatile<_Tp>::type;


  template<typename _Tp>
    using remove_cv_t = typename remove_cv<_Tp>::type;


  template<typename _Tp>
    using add_const_t = typename add_const<_Tp>::type;


  template<typename _Tp>
    using add_volatile_t = typename add_volatile<_Tp>::type;


  template<typename _Tp>
    using add_cv_t = typename add_cv<_Tp>::type;





  template<typename _Tp>
    struct remove_reference
    { typedef _Tp type; };

  template<typename _Tp>
    struct remove_reference<_Tp&>
    { typedef _Tp type; };

  template<typename _Tp>
    struct remove_reference<_Tp&&>
    { typedef _Tp type; };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __add_lvalue_reference_helper
    { typedef _Tp type; };

  template<typename _Tp>
    struct __add_lvalue_reference_helper<_Tp, true>
    { typedef _Tp& type; };


  template<typename _Tp>
    struct add_lvalue_reference
    : public __add_lvalue_reference_helper<_Tp>
    { };

  template<typename _Tp, bool = __is_referenceable<_Tp>::value>
    struct __add_rvalue_reference_helper
    { typedef _Tp type; };

  template<typename _Tp>
    struct __add_rvalue_reference_helper<_Tp, true>
    { typedef _Tp&& type; };


  template<typename _Tp>
    struct add_rvalue_reference
    : public __add_rvalue_reference_helper<_Tp>
    { };



  template<typename _Tp>
    using remove_reference_t = typename remove_reference<_Tp>::type;


  template<typename _Tp>
    using add_lvalue_reference_t = typename add_lvalue_reference<_Tp>::type;


  template<typename _Tp>
    using add_rvalue_reference_t = typename add_rvalue_reference<_Tp>::type;





  template<typename _Unqualified, bool _IsConst, bool _IsVol>
    struct __cv_selector;

  template<typename _Unqualified>
    struct __cv_selector<_Unqualified, false, false>
    { typedef _Unqualified __type; };

  template<typename _Unqualified>
    struct __cv_selector<_Unqualified, false, true>
    { typedef volatile _Unqualified __type; };

  template<typename _Unqualified>
    struct __cv_selector<_Unqualified, true, false>
    { typedef const _Unqualified __type; };

  template<typename _Unqualified>
    struct __cv_selector<_Unqualified, true, true>
    { typedef const volatile _Unqualified __type; };

  template<typename _Qualified, typename _Unqualified,
    bool _IsConst = is_const<_Qualified>::value,
    bool _IsVol = is_volatile<_Qualified>::value>
    class __match_cv_qualifiers
    {
      typedef __cv_selector<_Unqualified, _IsConst, _IsVol> __match;

    public:
      typedef typename __match::__type __type;
    };


  template<typename _Tp>
    struct __make_unsigned
    { typedef _Tp __type; };

  template<>
    struct __make_unsigned<char>
    { typedef unsigned char __type; };

  template<>
    struct __make_unsigned<signed char>
    { typedef unsigned char __type; };

  template<>
    struct __make_unsigned<short>
    { typedef unsigned short __type; };

  template<>
    struct __make_unsigned<int>
    { typedef unsigned int __type; };

  template<>
    struct __make_unsigned<long>
    { typedef unsigned long __type; };

  template<>
    struct __make_unsigned<long long>
    { typedef unsigned long long __type; };


  template<>
    struct __make_unsigned<__int128>
    { typedef unsigned __int128 __type; };
# 1730 "/usr/include/c++/10/type_traits" 3
  template<typename _Tp,
    bool _IsInt = is_integral<_Tp>::value,
    bool _IsEnum = is_enum<_Tp>::value>
    class __make_unsigned_selector;

  template<typename _Tp>
    class __make_unsigned_selector<_Tp, true, false>
    {
      using __unsigned_type
 = typename __make_unsigned<__remove_cv_t<_Tp>>::__type;

    public:
      using __type
 = typename __match_cv_qualifiers<_Tp, __unsigned_type>::__type;
    };

  class __make_unsigned_selector_base
  {
  protected:
    template<typename...> struct _List { };

    template<typename _Tp, typename... _Up>
      struct _List<_Tp, _Up...> : _List<_Up...>
      { static constexpr size_t __size = sizeof(_Tp); };

    template<size_t _Sz, typename _Tp, bool = (_Sz <= _Tp::__size)>
      struct __select;

    template<size_t _Sz, typename _Uint, typename... _UInts>
      struct __select<_Sz, _List<_Uint, _UInts...>, true>
      { using __type = _Uint; };

    template<size_t _Sz, typename _Uint, typename... _UInts>
      struct __select<_Sz, _List<_Uint, _UInts...>, false>
      : __select<_Sz, _List<_UInts...>>
      { };
  };


  template<typename _Tp>
    class __make_unsigned_selector<_Tp, false, true>
    : __make_unsigned_selector_base
    {

      using _UInts = _List<unsigned char, unsigned short, unsigned int,
      unsigned long, unsigned long long>;

      using __unsigned_type = typename __select<sizeof(_Tp), _UInts>::__type;

    public:
      using __type
 = typename __match_cv_qualifiers<_Tp, __unsigned_type>::__type;
    };






  template<>
    struct __make_unsigned<wchar_t>
    {
      using __type
 = typename __make_unsigned_selector<wchar_t, false, true>::__type;
    };
# 1806 "/usr/include/c++/10/type_traits" 3
  template<>
    struct __make_unsigned<char16_t>
    {
      using __type
 = typename __make_unsigned_selector<char16_t, false, true>::__type;
    };

  template<>
    struct __make_unsigned<char32_t>
    {
      using __type
 = typename __make_unsigned_selector<char32_t, false, true>::__type;
    };





  template<typename _Tp>
    struct make_unsigned
    { typedef typename __make_unsigned_selector<_Tp>::__type type; };


  template<>
    struct make_unsigned<bool>;



  template<typename _Tp>
    struct __make_signed
    { typedef _Tp __type; };

  template<>
    struct __make_signed<char>
    { typedef signed char __type; };

  template<>
    struct __make_signed<unsigned char>
    { typedef signed char __type; };

  template<>
    struct __make_signed<unsigned short>
    { typedef signed short __type; };

  template<>
    struct __make_signed<unsigned int>
    { typedef signed int __type; };

  template<>
    struct __make_signed<unsigned long>
    { typedef signed long __type; };

  template<>
    struct __make_signed<unsigned long long>
    { typedef signed long long __type; };


  template<>
    struct __make_signed<unsigned __int128>
    { typedef __int128 __type; };
# 1884 "/usr/include/c++/10/type_traits" 3
  template<typename _Tp,
    bool _IsInt = is_integral<_Tp>::value,
    bool _IsEnum = is_enum<_Tp>::value>
    class __make_signed_selector;

  template<typename _Tp>
    class __make_signed_selector<_Tp, true, false>
    {
      using __signed_type
 = typename __make_signed<__remove_cv_t<_Tp>>::__type;

    public:
      using __type
 = typename __match_cv_qualifiers<_Tp, __signed_type>::__type;
    };


  template<typename _Tp>
    class __make_signed_selector<_Tp, false, true>
    {
      typedef typename __make_unsigned_selector<_Tp>::__type __unsigned_type;

    public:
      typedef typename __make_signed_selector<__unsigned_type>::__type __type;
    };






  template<>
    struct __make_signed<wchar_t>
    {
      using __type
 = typename __make_signed_selector<wchar_t, false, true>::__type;
    };
# 1932 "/usr/include/c++/10/type_traits" 3
  template<>
    struct __make_signed<char16_t>
    {
      using __type
 = typename __make_signed_selector<char16_t, false, true>::__type;
    };

  template<>
    struct __make_signed<char32_t>
    {
      using __type
 = typename __make_signed_selector<char32_t, false, true>::__type;
    };





  template<typename _Tp>
    struct make_signed
    { typedef typename __make_signed_selector<_Tp>::__type type; };


  template<>
    struct make_signed<bool>;



  template<typename _Tp>
    using make_signed_t = typename make_signed<_Tp>::type;


  template<typename _Tp>
    using make_unsigned_t = typename make_unsigned<_Tp>::type;





  template<typename _Tp>
    struct remove_extent
    { typedef _Tp type; };

  template<typename _Tp, std::size_t _Size>
    struct remove_extent<_Tp[_Size]>
    { typedef _Tp type; };

  template<typename _Tp>
    struct remove_extent<_Tp[]>
    { typedef _Tp type; };


  template<typename _Tp>
    struct remove_all_extents
    { typedef _Tp type; };

  template<typename _Tp, std::size_t _Size>
    struct remove_all_extents<_Tp[_Size]>
    { typedef typename remove_all_extents<_Tp>::type type; };

  template<typename _Tp>
    struct remove_all_extents<_Tp[]>
    { typedef typename remove_all_extents<_Tp>::type type; };



  template<typename _Tp>
    using remove_extent_t = typename remove_extent<_Tp>::type;


  template<typename _Tp>
    using remove_all_extents_t = typename remove_all_extents<_Tp>::type;




  template<typename _Tp, typename>
    struct __remove_pointer_helper
    { typedef _Tp type; };

  template<typename _Tp, typename _Up>
    struct __remove_pointer_helper<_Tp, _Up*>
    { typedef _Up type; };


  template<typename _Tp>
    struct remove_pointer
    : public __remove_pointer_helper<_Tp, __remove_cv_t<_Tp>>
    { };


  template<typename _Tp, bool = __or_<__is_referenceable<_Tp>,
          is_void<_Tp>>::value>
    struct __add_pointer_helper
    { typedef _Tp type; };

  template<typename _Tp>
    struct __add_pointer_helper<_Tp, true>
    { typedef typename remove_reference<_Tp>::type* type; };

  template<typename _Tp>
    struct add_pointer
    : public __add_pointer_helper<_Tp>
    { };



  template<typename _Tp>
    using remove_pointer_t = typename remove_pointer<_Tp>::type;


  template<typename _Tp>
    using add_pointer_t = typename add_pointer<_Tp>::type;


  template<std::size_t _Len>
    struct __aligned_storage_msa
    {
      union __type
      {
 unsigned char __data[_Len];
 struct __attribute__((__aligned__)) { } __align;
      };
    };
# 2067 "/usr/include/c++/10/type_traits" 3
  template<std::size_t _Len, std::size_t _Align =
    __alignof__(typename __aligned_storage_msa<_Len>::__type)>
    struct aligned_storage
    {
      union type
      {
 unsigned char __data[_Len];
 struct __attribute__((__aligned__((_Align)))) { } __align;
      };
    };

  template <typename... _Types>
    struct __strictest_alignment
    {
      static const size_t _S_alignment = 0;
      static const size_t _S_size = 0;
    };

  template <typename _Tp, typename... _Types>
    struct __strictest_alignment<_Tp, _Types...>
    {
      static const size_t _S_alignment =
        alignof(_Tp) > __strictest_alignment<_Types...>::_S_alignment
 ? alignof(_Tp) : __strictest_alignment<_Types...>::_S_alignment;
      static const size_t _S_size =
        sizeof(_Tp) > __strictest_alignment<_Types...>::_S_size
 ? sizeof(_Tp) : __strictest_alignment<_Types...>::_S_size;
    };
# 2106 "/usr/include/c++/10/type_traits" 3
  template <size_t _Len, typename... _Types>
    struct aligned_union
    {
    private:
      static_assert(sizeof...(_Types) != 0, "At least one type is required");

      using __strictest = __strictest_alignment<_Types...>;
      static const size_t _S_len = _Len > __strictest::_S_size
 ? _Len : __strictest::_S_size;
    public:

      static const size_t alignment_value = __strictest::_S_alignment;

      typedef typename aligned_storage<_S_len, alignment_value>::type type;
    };

  template <size_t _Len, typename... _Types>
    const size_t aligned_union<_Len, _Types...>::alignment_value;



  template<typename _Up,
    bool _IsArray = is_array<_Up>::value,
    bool _IsFunction = is_function<_Up>::value>
    struct __decay_selector;


  template<typename _Up>
    struct __decay_selector<_Up, false, false>
    { typedef __remove_cv_t<_Up> __type; };

  template<typename _Up>
    struct __decay_selector<_Up, true, false>
    { typedef typename remove_extent<_Up>::type* __type; };

  template<typename _Up>
    struct __decay_selector<_Up, false, true>
    { typedef typename add_pointer<_Up>::type __type; };


  template<typename _Tp>
    class decay
    {
      typedef typename remove_reference<_Tp>::type __remove_type;

    public:
      typedef typename __decay_selector<__remove_type>::__type type;
    };


  template<typename _Tp>
    using __decay_t = typename decay<_Tp>::type;

  template<typename _Tp>
    class reference_wrapper;


  template<typename _Tp>
    struct __strip_reference_wrapper
    {
      typedef _Tp __type;
    };

  template<typename _Tp>
    struct __strip_reference_wrapper<reference_wrapper<_Tp> >
    {
      typedef _Tp& __type;
    };

  template<typename _Tp>
    using __decay_and_strip = __strip_reference_wrapper<__decay_t<_Tp>>;




  template<bool, typename _Tp = void>
    struct enable_if
    { };


  template<typename _Tp>
    struct enable_if<true, _Tp>
    { typedef _Tp type; };


  template<bool _Cond, typename _Tp = void>
    using __enable_if_t = typename enable_if<_Cond, _Tp>::type;

  template<typename... _Cond>
    using _Require = __enable_if_t<__and_<_Cond...>::value>;



  template<bool _Cond, typename _Iftrue, typename _Iffalse>
    struct conditional
    { typedef _Iftrue type; };


  template<typename _Iftrue, typename _Iffalse>
    struct conditional<false, _Iftrue, _Iffalse>
    { typedef _Iffalse type; };


  template<typename _Tp>
    using __remove_cvref_t
     = typename remove_cv<typename remove_reference<_Tp>::type>::type;


  template<typename... _Tp>
    struct common_type;



  struct __do_common_type_impl
  {
    template<typename _Tp, typename _Up>
      using __cond_t
 = decltype(true ? std::declval<_Tp>() : std::declval<_Up>());



    template<typename _Tp, typename _Up>
      static __success_type<__decay_t<__cond_t<_Tp, _Up>>>
      _S_test(int);
# 2239 "/usr/include/c++/10/type_traits" 3
    template<typename, typename>
      static __failure_type
      _S_test_2(...);

    template<typename _Tp, typename _Up>
      static decltype(_S_test_2<_Tp, _Up>(0))
      _S_test(...);
  };


  template<>
    struct common_type<>
    { };


  template<typename _Tp0>
    struct common_type<_Tp0>
    : public common_type<_Tp0, _Tp0>
    { };


  template<typename _Tp1, typename _Tp2,
    typename _Dp1 = __decay_t<_Tp1>, typename _Dp2 = __decay_t<_Tp2>>
    struct __common_type_impl
    {


      using type = common_type<_Dp1, _Dp2>;
    };

  template<typename _Tp1, typename _Tp2>
    struct __common_type_impl<_Tp1, _Tp2, _Tp1, _Tp2>
    : private __do_common_type_impl
    {


      using type = decltype(_S_test<_Tp1, _Tp2>(0));
    };


  template<typename _Tp1, typename _Tp2>
    struct common_type<_Tp1, _Tp2>
    : public __common_type_impl<_Tp1, _Tp2>::type
    { };

  template<typename...>
    struct __common_type_pack
    { };

  template<typename, typename, typename = void>
    struct __common_type_fold;


  template<typename _Tp1, typename _Tp2, typename... _Rp>
    struct common_type<_Tp1, _Tp2, _Rp...>
    : public __common_type_fold<common_type<_Tp1, _Tp2>,
    __common_type_pack<_Rp...>>
    { };




  template<typename _CTp, typename... _Rp>
    struct __common_type_fold<_CTp, __common_type_pack<_Rp...>,
         __void_t<typename _CTp::type>>
    : public common_type<typename _CTp::type, _Rp...>
    { };


  template<typename _CTp, typename _Rp>
    struct __common_type_fold<_CTp, _Rp, void>
    { };

  template<typename _Tp, bool = is_enum<_Tp>::value>
    struct __underlying_type_impl
    {
      using type = __underlying_type(_Tp);
    };

  template<typename _Tp>
    struct __underlying_type_impl<_Tp, false>
    { };


  template<typename _Tp>
    struct underlying_type
    : public __underlying_type_impl<_Tp>
    { };

  template<typename _Tp>
    struct __declval_protector
    {
      static const bool __stop = false;
    };

  template<typename _Tp>
    auto declval() noexcept -> decltype(__declval<_Tp>(0))
    {
      static_assert(__declval_protector<_Tp>::__stop,
      "declval() must not be used!");
      return __declval<_Tp>(0);
    }


  template<typename _Signature>
    class result_of;





  struct __invoke_memfun_ref { };
  struct __invoke_memfun_deref { };
  struct __invoke_memobj_ref { };
  struct __invoke_memobj_deref { };
  struct __invoke_other { };


  template<typename _Tp, typename _Tag>
    struct __result_of_success : __success_type<_Tp>
    { using __invoke_type = _Tag; };


  struct __result_of_memfun_ref_impl
  {
    template<typename _Fp, typename _Tp1, typename... _Args>
      static __result_of_success<decltype(
      (std::declval<_Tp1>().*std::declval<_Fp>())(std::declval<_Args>()...)
      ), __invoke_memfun_ref> _S_test(int);

    template<typename...>
      static __failure_type _S_test(...);
  };

  template<typename _MemPtr, typename _Arg, typename... _Args>
    struct __result_of_memfun_ref
    : private __result_of_memfun_ref_impl
    {
      typedef decltype(_S_test<_MemPtr, _Arg, _Args...>(0)) type;
    };


  struct __result_of_memfun_deref_impl
  {
    template<typename _Fp, typename _Tp1, typename... _Args>
      static __result_of_success<decltype(
      ((*std::declval<_Tp1>()).*std::declval<_Fp>())(std::declval<_Args>()...)
      ), __invoke_memfun_deref> _S_test(int);

    template<typename...>
      static __failure_type _S_test(...);
  };

  template<typename _MemPtr, typename _Arg, typename... _Args>
    struct __result_of_memfun_deref
    : private __result_of_memfun_deref_impl
    {
      typedef decltype(_S_test<_MemPtr, _Arg, _Args...>(0)) type;
    };


  struct __result_of_memobj_ref_impl
  {
    template<typename _Fp, typename _Tp1>
      static __result_of_success<decltype(
      std::declval<_Tp1>().*std::declval<_Fp>()
      ), __invoke_memobj_ref> _S_test(int);

    template<typename, typename>
      static __failure_type _S_test(...);
  };

  template<typename _MemPtr, typename _Arg>
    struct __result_of_memobj_ref
    : private __result_of_memobj_ref_impl
    {
      typedef decltype(_S_test<_MemPtr, _Arg>(0)) type;
    };


  struct __result_of_memobj_deref_impl
  {
    template<typename _Fp, typename _Tp1>
      static __result_of_success<decltype(
      (*std::declval<_Tp1>()).*std::declval<_Fp>()
      ), __invoke_memobj_deref> _S_test(int);

    template<typename, typename>
      static __failure_type _S_test(...);
  };

  template<typename _MemPtr, typename _Arg>
    struct __result_of_memobj_deref
    : private __result_of_memobj_deref_impl
    {
      typedef decltype(_S_test<_MemPtr, _Arg>(0)) type;
    };

  template<typename _MemPtr, typename _Arg>
    struct __result_of_memobj;

  template<typename _Res, typename _Class, typename _Arg>
    struct __result_of_memobj<_Res _Class::*, _Arg>
    {
      typedef __remove_cvref_t<_Arg> _Argval;
      typedef _Res _Class::* _MemPtr;
      typedef typename conditional<__or_<is_same<_Argval, _Class>,
        is_base_of<_Class, _Argval>>::value,
        __result_of_memobj_ref<_MemPtr, _Arg>,
        __result_of_memobj_deref<_MemPtr, _Arg>
      >::type::type type;
    };

  template<typename _MemPtr, typename _Arg, typename... _Args>
    struct __result_of_memfun;

  template<typename _Res, typename _Class, typename _Arg, typename... _Args>
    struct __result_of_memfun<_Res _Class::*, _Arg, _Args...>
    {
      typedef typename remove_reference<_Arg>::type _Argval;
      typedef _Res _Class::* _MemPtr;
      typedef typename conditional<is_base_of<_Class, _Argval>::value,
        __result_of_memfun_ref<_MemPtr, _Arg, _Args...>,
        __result_of_memfun_deref<_MemPtr, _Arg, _Args...>
      >::type::type type;
    };






  template<typename _Tp, typename _Up = __remove_cvref_t<_Tp>>
    struct __inv_unwrap
    {
      using type = _Tp;
    };

  template<typename _Tp, typename _Up>
    struct __inv_unwrap<_Tp, reference_wrapper<_Up>>
    {
      using type = _Up&;
    };

  template<bool, bool, typename _Functor, typename... _ArgTypes>
    struct __result_of_impl
    {
      typedef __failure_type type;
    };

  template<typename _MemPtr, typename _Arg>
    struct __result_of_impl<true, false, _MemPtr, _Arg>
    : public __result_of_memobj<__decay_t<_MemPtr>,
    typename __inv_unwrap<_Arg>::type>
    { };

  template<typename _MemPtr, typename _Arg, typename... _Args>
    struct __result_of_impl<false, true, _MemPtr, _Arg, _Args...>
    : public __result_of_memfun<__decay_t<_MemPtr>,
    typename __inv_unwrap<_Arg>::type, _Args...>
    { };


  struct __result_of_other_impl
  {
    template<typename _Fn, typename... _Args>
      static __result_of_success<decltype(
      std::declval<_Fn>()(std::declval<_Args>()...)
      ), __invoke_other> _S_test(int);

    template<typename...>
      static __failure_type _S_test(...);
  };

  template<typename _Functor, typename... _ArgTypes>
    struct __result_of_impl<false, false, _Functor, _ArgTypes...>
    : private __result_of_other_impl
    {
      typedef decltype(_S_test<_Functor, _ArgTypes...>(0)) type;
    };


  template<typename _Functor, typename... _ArgTypes>
    struct __invoke_result
    : public __result_of_impl<
        is_member_object_pointer<
          typename remove_reference<_Functor>::type
        >::value,
        is_member_function_pointer<
          typename remove_reference<_Functor>::type
        >::value,
 _Functor, _ArgTypes...
      >::type
    { };

  template<typename _Functor, typename... _ArgTypes>
    struct result_of<_Functor(_ArgTypes...)>
    : public __invoke_result<_Functor, _ArgTypes...>
    { };



  template<size_t _Len, size_t _Align =
     __alignof__(typename __aligned_storage_msa<_Len>::__type)>
    using aligned_storage_t = typename aligned_storage<_Len, _Align>::type;

  template <size_t _Len, typename... _Types>
    using aligned_union_t = typename aligned_union<_Len, _Types...>::type;


  template<typename _Tp>
    using decay_t = typename decay<_Tp>::type;


  template<bool _Cond, typename _Tp = void>
    using enable_if_t = typename enable_if<_Cond, _Tp>::type;


  template<bool _Cond, typename _Iftrue, typename _Iffalse>
    using conditional_t = typename conditional<_Cond, _Iftrue, _Iffalse>::type;


  template<typename... _Tp>
    using common_type_t = typename common_type<_Tp...>::type;


  template<typename _Tp>
    using underlying_type_t = typename underlying_type<_Tp>::type;


  template<typename _Tp>
    using result_of_t = typename result_of<_Tp>::type;





  template<typename...> using void_t = void;



  template<typename _Default, typename _AlwaysVoid,
    template<typename...> class _Op, typename... _Args>
    struct __detector
    {
      using value_t = false_type;
      using type = _Default;
    };


  template<typename _Default, template<typename...> class _Op,
     typename... _Args>
    struct __detector<_Default, __void_t<_Op<_Args...>>, _Op, _Args...>
    {
      using value_t = true_type;
      using type = _Op<_Args...>;
    };


  template<typename _Default, template<typename...> class _Op,
    typename... _Args>
    using __detected_or = __detector<_Default, void, _Op, _Args...>;


  template<typename _Default, template<typename...> class _Op,
    typename... _Args>
    using __detected_or_t
      = typename __detected_or<_Default, _Op, _Args...>::type;
# 2624 "/usr/include/c++/10/type_traits" 3
  template <typename _Tp>
    struct __is_swappable;

  template <typename _Tp>
    struct __is_nothrow_swappable;

  template<typename... _Elements>
    class tuple;

  template<typename>
    struct __is_tuple_like_impl : false_type
    { };

  template<typename... _Tps>
    struct __is_tuple_like_impl<tuple<_Tps...>> : true_type
    { };


  template<typename _Tp>
    struct __is_tuple_like
    : public __is_tuple_like_impl<__remove_cvref_t<_Tp>>::type
    { };

  template<typename _Tp>
   
    inline
    _Require<__not_<__is_tuple_like<_Tp>>,
      is_move_constructible<_Tp>,
      is_move_assignable<_Tp>>
    swap(_Tp&, _Tp&)
    noexcept(__and_<is_nothrow_move_constructible<_Tp>,
             is_nothrow_move_assignable<_Tp>>::value);

  template<typename _Tp, size_t _Nm>
   
    inline
    __enable_if_t<__is_swappable<_Tp>::value>
    swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm])
    noexcept(__is_nothrow_swappable<_Tp>::value);

  namespace __swappable_details {
    using std::swap;

    struct __do_is_swappable_impl
    {
      template<typename _Tp, typename
               = decltype(swap(std::declval<_Tp&>(), std::declval<_Tp&>()))>
        static true_type __test(int);

      template<typename>
        static false_type __test(...);
    };

    struct __do_is_nothrow_swappable_impl
    {
      template<typename _Tp>
        static __bool_constant<
          noexcept(swap(std::declval<_Tp&>(), std::declval<_Tp&>()))
        > __test(int);

      template<typename>
        static false_type __test(...);
    };

  }

  template<typename _Tp>
    struct __is_swappable_impl
    : public __swappable_details::__do_is_swappable_impl
    {
      typedef decltype(__test<_Tp>(0)) type;
    };

  template<typename _Tp>
    struct __is_nothrow_swappable_impl
    : public __swappable_details::__do_is_nothrow_swappable_impl
    {
      typedef decltype(__test<_Tp>(0)) type;
    };

  template<typename _Tp>
    struct __is_swappable
    : public __is_swappable_impl<_Tp>::type
    { };

  template<typename _Tp>
    struct __is_nothrow_swappable
    : public __is_nothrow_swappable_impl<_Tp>::type
    { };






  template<typename _Tp>
    struct is_swappable
    : public __is_swappable_impl<_Tp>::type
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };


  template<typename _Tp>
    struct is_nothrow_swappable
    : public __is_nothrow_swappable_impl<_Tp>::type
    {
      static_assert(std::__is_complete_or_unbounded(__type_identity<_Tp>{}),
 "template argument must be a complete class or an unbounded array");
    };



  template<typename _Tp>
    constexpr bool is_swappable_v =
      is_swappable<_Tp>::value;


  template<typename _Tp>
    constexpr bool is_nothrow_swappable_v =
      is_nothrow_swappable<_Tp>::value;


  namespace __swappable_with_details {
    using std::swap;

    struct __do_is_swappable_with_impl
    {
      template<typename _Tp, typename _Up, typename
               = decltype(swap(std::declval<_Tp>(), std::declval<_Up>())),
               typename
               = decltype(swap(std::declval<_Up>(), std::declval<_Tp>()))>
        static true_type __test(int);

      template<typename, typename>
        static false_type __test(...);
    };

    struct __do_is_nothrow_swappable_with_impl
    {
      template<typename _Tp, typename _Up>
        static __bool_constant<
          noexcept(swap(std::declval<_Tp>(), std::declval<_Up>()))
          &&
          noexcept(swap(std::declval<_Up>(), std::declval<_Tp>()))
        > __test(int);

      template<typename, typename>
        static false_type __test(...);
    };

  }

  template<typename _Tp, typename _Up>
    struct __is_swappable_with_impl
    : public __swappable_with_details::__do_is_swappable_with_impl
    {
      typedef decltype(__test<_Tp, _Up>(0)) type;
    };


  template<typename _Tp>
    struct __is_swappable_with_impl<_Tp&, _Tp&>
    : public __swappable_details::__do_is_swappable_impl
    {
      typedef decltype(__test<_Tp&>(0)) type;
    };

  template<typename _Tp, typename _Up>
    struct __is_nothrow_swappable_with_impl
    : public __swappable_with_details::__do_is_nothrow_swappable_with_impl
    {
      typedef decltype(__test<_Tp, _Up>(0)) type;
    };


  template<typename _Tp>
    struct __is_nothrow_swappable_with_impl<_Tp&, _Tp&>
    : public __swappable_details::__do_is_nothrow_swappable_impl
    {
      typedef decltype(__test<_Tp&>(0)) type;
    };


  template<typename _Tp, typename _Up>
    struct is_swappable_with
    : public __is_swappable_with_impl<_Tp, _Up>::type
    { };


  template<typename _Tp, typename _Up>
    struct is_nothrow_swappable_with
    : public __is_nothrow_swappable_with_impl<_Tp, _Up>::type
    { };



  template<typename _Tp, typename _Up>
    constexpr bool is_swappable_with_v =
      is_swappable_with<_Tp, _Up>::value;


  template<typename _Tp, typename _Up>
    constexpr bool is_nothrow_swappable_with_v =
      is_nothrow_swappable_with<_Tp, _Up>::value;







  template<typename _Result, typename _Ret,
    bool = is_void<_Ret>::value, typename = void>
    struct __is_invocable_impl : false_type { };


  template<typename _Result, typename _Ret>
    struct __is_invocable_impl<_Result, _Ret,
                                true,
          __void_t<typename _Result::type>>
    : true_type
    { };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"

  template<typename _Result, typename _Ret>
    struct __is_invocable_impl<_Result, _Ret,
                                false,
          __void_t<typename _Result::type>>
    {
    private:


      static typename _Result::type _S_get();

      template<typename _Tp>
 static void _S_conv(_Tp);


      template<typename _Tp, typename = decltype(_S_conv<_Tp>(_S_get()))>
 static true_type
 _S_test(int);

      template<typename _Tp>
 static false_type
 _S_test(...);

    public:
      using type = decltype(_S_test<_Ret>(1));
    };
#pragma GCC diagnostic pop

  template<typename _Fn, typename... _ArgTypes>
    struct __is_invocable
    : __is_invocable_impl<__invoke_result<_Fn, _ArgTypes...>, void>::type
    { };

  template<typename _Fn, typename _Tp, typename... _Args>
    constexpr bool __call_is_nt(__invoke_memfun_ref)
    {
      using _Up = typename __inv_unwrap<_Tp>::type;
      return noexcept((std::declval<_Up>().*std::declval<_Fn>())(
     std::declval<_Args>()...));
    }

  template<typename _Fn, typename _Tp, typename... _Args>
    constexpr bool __call_is_nt(__invoke_memfun_deref)
    {
      return noexcept(((*std::declval<_Tp>()).*std::declval<_Fn>())(
     std::declval<_Args>()...));
    }

  template<typename _Fn, typename _Tp>
    constexpr bool __call_is_nt(__invoke_memobj_ref)
    {
      using _Up = typename __inv_unwrap<_Tp>::type;
      return noexcept(std::declval<_Up>().*std::declval<_Fn>());
    }

  template<typename _Fn, typename _Tp>
    constexpr bool __call_is_nt(__invoke_memobj_deref)
    {
      return noexcept((*std::declval<_Tp>()).*std::declval<_Fn>());
    }

  template<typename _Fn, typename... _Args>
    constexpr bool __call_is_nt(__invoke_other)
    {
      return noexcept(std::declval<_Fn>()(std::declval<_Args>()...));
    }

  template<typename _Result, typename _Fn, typename... _Args>
    struct __call_is_nothrow
    : __bool_constant<
 std::__call_is_nt<_Fn, _Args...>(typename _Result::__invoke_type{})
      >
    { };

  template<typename _Fn, typename... _Args>
    using __call_is_nothrow_
      = __call_is_nothrow<__invoke_result<_Fn, _Args...>, _Fn, _Args...>;


  template<typename _Fn, typename... _Args>
    struct __is_nothrow_invocable
    : __and_<__is_invocable<_Fn, _Args...>,
             __call_is_nothrow_<_Fn, _Args...>>::type
    { };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
  struct __nonesuchbase {};
  struct __nonesuch : private __nonesuchbase {
    ~__nonesuch() = delete;
    __nonesuch(__nonesuch const&) = delete;
    void operator=(__nonesuch const&) = delete;
  };
#pragma GCC diagnostic pop
# 3455 "/usr/include/c++/10/type_traits" 3

}
# 58 "/usr/include/c++/10/bits/move.h" 2 3

namespace std __attribute__ ((__visibility__ ("default")))
{

# 74 "/usr/include/c++/10/bits/move.h" 3
  template<typename _Tp>
    constexpr _Tp&&
    forward(typename std::remove_reference<_Tp>::type& __t) noexcept
    { return static_cast<_Tp&&>(__t); }







  template<typename _Tp>
    constexpr _Tp&&
    forward(typename std::remove_reference<_Tp>::type&& __t) noexcept
    {
      static_assert(!std::is_lvalue_reference<_Tp>::value, "template argument"
      " substituting _Tp is an lvalue reference type");
      return static_cast<_Tp&&>(__t);
    }






  template<typename _Tp>
    constexpr typename std::remove_reference<_Tp>::type&&
    move(_Tp&& __t) noexcept
    { return static_cast<typename std::remove_reference<_Tp>::type&&>(__t); }


  template<typename _Tp>
    struct __move_if_noexcept_cond
    : public __and_<__not_<is_nothrow_move_constructible<_Tp>>,
                    is_copy_constructible<_Tp>>::type { };
# 118 "/usr/include/c++/10/bits/move.h" 3
  template<typename _Tp>
    constexpr typename
    conditional<__move_if_noexcept_cond<_Tp>::value, const _Tp&, _Tp&&>::type
    move_if_noexcept(_Tp& __x) noexcept
    { return std::move(__x); }
# 138 "/usr/include/c++/10/bits/move.h" 3
  template<typename _Tp>
    inline _Tp*
    addressof(_Tp& __r) noexcept
    { return std::__addressof(__r); }



  template<typename _Tp>
    const _Tp* addressof(const _Tp&&) = delete;


  template <typename _Tp, typename _Up = _Tp>
   
    inline _Tp
    __exchange(_Tp& __obj, _Up&& __new_val)
    {
      _Tp __old_val = std::move(__obj);
      __obj = std::forward<_Up>(__new_val);
      return __old_val;
    }
# 179 "/usr/include/c++/10/bits/move.h" 3
  template<typename _Tp>
   
    inline

    typename enable_if<__and_<__not_<__is_tuple_like<_Tp>>,
         is_move_constructible<_Tp>,
         is_move_assignable<_Tp>>::value>::type



    swap(_Tp& __a, _Tp& __b)
    noexcept(__and_<is_nothrow_move_constructible<_Tp>, is_nothrow_move_assignable<_Tp>>::value)

    {




      _Tp __tmp = std::move(__a);
      __a = std::move(__b);
      __b = std::move(__tmp);
    }




  template<typename _Tp, size_t _Nm>
   
    inline

    typename enable_if<__is_swappable<_Tp>::value>::type



    swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm])
    noexcept(__is_nothrow_swappable<_Tp>::value)
    {
      for (size_t __n = 0; __n < _Nm; ++__n)
 swap(__a[__n], __b[__n]);
    }



}
# 41 "/usr/include/c++/10/bits/nested_exception.h" 2 3

extern "C++" {

namespace std
{






  class nested_exception
  {
    exception_ptr _M_ptr;

  public:
    nested_exception() noexcept : _M_ptr(current_exception()) { }

    nested_exception(const nested_exception&) noexcept = default;

    nested_exception& operator=(const nested_exception&) noexcept = default;

    virtual ~nested_exception() noexcept;

    [[noreturn]]
    void
    rethrow_nested() const
    {
      if (_M_ptr)
 rethrow_exception(_M_ptr);
      std::terminate();
    }

    exception_ptr
    nested_ptr() const noexcept
    { return _M_ptr; }
  };



  template<typename _Except>
    struct _Nested_exception : public _Except, public nested_exception
    {
      explicit _Nested_exception(const _Except& __ex)
      : _Except(__ex)
      { }

      explicit _Nested_exception(_Except&& __ex)
      : _Except(static_cast<_Except&&>(__ex))
      { }
    };




  template<typename _Tp>
    [[noreturn]]
    inline void
    __throw_with_nested_impl(_Tp&& __t, true_type)
    {
      using _Up = typename remove_reference<_Tp>::type;
      throw _Nested_exception<_Up>{std::forward<_Tp>(__t)};
    }

  template<typename _Tp>
    [[noreturn]]
    inline void
    __throw_with_nested_impl(_Tp&& __t, false_type)
    { throw std::forward<_Tp>(__t); }





  template<typename _Tp>
    [[noreturn]]
    inline void
    throw_with_nested(_Tp&& __t)
    {
      using _Up = typename decay<_Tp>::type;
      using _CopyConstructible
 = __and_<is_copy_constructible<_Up>, is_move_constructible<_Up>>;
      static_assert(_CopyConstructible::value,
   "throw_with_nested argument must be CopyConstructible");
      using __nest = __and_<is_class<_Up>, __bool_constant<!__is_final(_Up)>,
       __not_<is_base_of<nested_exception, _Up>>>;
      std::__throw_with_nested_impl(std::forward<_Tp>(__t), __nest{});
    }




  template<typename _Tp>
    using __rethrow_if_nested_cond = typename enable_if<
      __and_<is_polymorphic<_Tp>,
      __or_<__not_<is_base_of<nested_exception, _Tp>>,
     is_convertible<_Tp*, nested_exception*>>>::value
    >::type;


  template<typename _Ex>
    inline __rethrow_if_nested_cond<_Ex>
    __rethrow_if_nested_impl(const _Ex* __ptr)
    {
      if (auto __ne_ptr = dynamic_cast<const nested_exception*>(__ptr))
 __ne_ptr->rethrow_nested();
    }


  inline void
  __rethrow_if_nested_impl(const void*)
  { }




  template<typename _Ex>
    inline void
    rethrow_if_nested(const _Ex& __ex)
    { std::__rethrow_if_nested_impl(std::__addressof(__ex)); }


}

}



#pragma GCC visibility pop
# 149 "/usr/include/c++/10/exception" 2 3
# 42 "/usr/include/c++/10/new" 2 3

#pragma GCC visibility push(default)

extern "C++" {

namespace std
{






  class bad_alloc : public exception
  {
  public:
    bad_alloc() throw() { }


    bad_alloc(const bad_alloc&) = default;
    bad_alloc& operator=(const bad_alloc&) = default;




    virtual ~bad_alloc() throw();


    virtual const char* what() const throw();
  };


  class bad_array_new_length : public bad_alloc
  {
  public:
    bad_array_new_length() throw() { }



    virtual ~bad_array_new_length() throw();


    virtual const char* what() const throw();
  };






  struct nothrow_t
  {

    explicit nothrow_t() = default;

  };

  extern const nothrow_t nothrow;



  typedef void (*new_handler)();



  new_handler set_new_handler(new_handler) throw();



  new_handler get_new_handler() noexcept;

}
# 126 "/usr/include/c++/10/new" 3
 void* operator new(std::size_t)
  __attribute__((__externally_visible__));
 void* operator new[](std::size_t)
  __attribute__((__externally_visible__));
void operator delete(void*) noexcept
  __attribute__((__externally_visible__));
void operator delete[](void*) noexcept
  __attribute__((__externally_visible__));

void operator delete(void*, std::size_t) noexcept
  __attribute__((__externally_visible__));
void operator delete[](void*, std::size_t) noexcept
  __attribute__((__externally_visible__));

 void* operator new(std::size_t, const std::nothrow_t&) noexcept
  __attribute__((__externally_visible__, __malloc__));
 void* operator new[](std::size_t, const std::nothrow_t&) noexcept
  __attribute__((__externally_visible__, __malloc__));
void operator delete(void*, const std::nothrow_t&) noexcept
  __attribute__((__externally_visible__));
void operator delete[](void*, const std::nothrow_t&) noexcept
  __attribute__((__externally_visible__));
# 174 "/usr/include/c++/10/new" 3
 inline void* operator new(std::size_t, void* __p) noexcept
{ return __p; }
 inline void* operator new[](std::size_t, void* __p) noexcept
{ return __p; }


inline void operator delete (void*, void*) noexcept { }
inline void operator delete[](void*, void*) noexcept { }

}
# 230 "/usr/include/c++/10/new" 3
#pragma GCC visibility pop
# 112 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 2
# 125 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"

# 125 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void* operator new(std:: size_t, void*) throw();
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void* operator new[](std:: size_t, void*) throw();
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete(void*, void*) throw();
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete[](void*, void*) throw();

extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete(void*, std:: size_t) throw();
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete[](void*, std:: size_t) throw();




# 1 "/usr/include/stdio.h" 1 3 4
# 27 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 28 "/usr/include/stdio.h" 2 3 4


# 29 "/usr/include/stdio.h" 3 4
extern "C" {



# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 1 3 4
# 34 "/usr/include/stdio.h" 2 3 4


# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stdarg.h" 1 3 4
# 40 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stdarg.h" 3 4
typedef __builtin_va_list __gnuc_va_list;
# 37 "/usr/include/stdio.h" 2 3 4


# 1 "/usr/include/bits/types/__fpos_t.h" 1 3 4




# 1 "/usr/include/bits/types/__mbstate_t.h" 1 3 4
# 13 "/usr/include/bits/types/__mbstate_t.h" 3 4
typedef struct
{
  int __count;
  union
  {
    unsigned int __wch;
    char __wchb[4];
  } __value;
} __mbstate_t;
# 6 "/usr/include/bits/types/__fpos_t.h" 2 3 4




typedef struct _G_fpos_t
{
  __off_t __pos;
  __mbstate_t __state;
} __fpos_t;
# 40 "/usr/include/stdio.h" 2 3 4
# 1 "/usr/include/bits/types/__fpos64_t.h" 1 3 4
# 10 "/usr/include/bits/types/__fpos64_t.h" 3 4
typedef struct _G_fpos64_t
{
  __off64_t __pos;
  __mbstate_t __state;
} __fpos64_t;
# 41 "/usr/include/stdio.h" 2 3 4
# 1 "/usr/include/bits/types/__FILE.h" 1 3 4



struct _IO_FILE;
typedef struct _IO_FILE __FILE;
# 42 "/usr/include/stdio.h" 2 3 4
# 1 "/usr/include/bits/types/FILE.h" 1 3 4



struct _IO_FILE;


typedef struct _IO_FILE FILE;
# 43 "/usr/include/stdio.h" 2 3 4
# 1 "/usr/include/bits/types/struct_FILE.h" 1 3 4
# 35 "/usr/include/bits/types/struct_FILE.h" 3 4
struct _IO_FILE;
struct _IO_marker;
struct _IO_codecvt;
struct _IO_wide_data;




typedef void _IO_lock_t;





struct _IO_FILE
{
  int _flags;


  char *_IO_read_ptr;
  char *_IO_read_end;
  char *_IO_read_base;
  char *_IO_write_base;
  char *_IO_write_ptr;
  char *_IO_write_end;
  char *_IO_buf_base;
  char *_IO_buf_end;


  char *_IO_save_base;
  char *_IO_backup_base;
  char *_IO_save_end;

  struct _IO_marker *_markers;

  struct _IO_FILE *_chain;

  int _fileno;
  int _flags2;
  __off_t _old_offset;


  unsigned short _cur_column;
  signed char _vtable_offset;
  char _shortbuf[1];

  _IO_lock_t *_lock;







  __off64_t _offset;

  struct _IO_codecvt *_codecvt;
  struct _IO_wide_data *_wide_data;
  struct _IO_FILE *_freeres_list;
  void *_freeres_buf;
  size_t __pad5;
  int _mode;

  char _unused2[15 * sizeof (int) - 4 * sizeof (void *) - sizeof (size_t)];
};
# 44 "/usr/include/stdio.h" 2 3 4


# 1 "/usr/include/bits/types/cookie_io_functions_t.h" 1 3 4
# 27 "/usr/include/bits/types/cookie_io_functions_t.h" 3 4
typedef __ssize_t cookie_read_function_t (void *__cookie, char *__buf,
                                          size_t __nbytes);







typedef __ssize_t cookie_write_function_t (void *__cookie, const char *__buf,
                                           size_t __nbytes);







typedef int cookie_seek_function_t (void *__cookie, __off64_t *__pos, int __w);


typedef int cookie_close_function_t (void *__cookie);






typedef struct _IO_cookie_io_functions_t
{
  cookie_read_function_t *read;
  cookie_write_function_t *write;
  cookie_seek_function_t *seek;
  cookie_close_function_t *close;
} cookie_io_functions_t;
# 47 "/usr/include/stdio.h" 2 3 4





typedef __gnuc_va_list va_list;
# 63 "/usr/include/stdio.h" 3 4
typedef __off_t off_t;






typedef __off64_t off64_t;






typedef __ssize_t ssize_t;






typedef __fpos_t fpos_t;




typedef __fpos64_t fpos64_t;
# 133 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/bits/stdio_lim.h" 1 3 4
# 134 "/usr/include/stdio.h" 2 3 4



extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;






extern int remove (const char *__filename) throw ();

extern int rename (const char *__old, const char *__new) throw ();



extern int renameat (int __oldfd, const char *__old, int __newfd,
       const char *__new) throw ();
# 164 "/usr/include/stdio.h" 3 4
extern int renameat2 (int __oldfd, const char *__old, int __newfd,
        const char *__new, unsigned int __flags) throw ();







extern FILE *tmpfile (void) ;
# 183 "/usr/include/stdio.h" 3 4
extern FILE *tmpfile64 (void) ;



extern char *tmpnam (char *__s) throw () ;




extern char *tmpnam_r (char *__s) throw () ;
# 204 "/usr/include/stdio.h" 3 4
extern char *tempnam (const char *__dir, const char *__pfx)
     throw () __attribute__ ((__malloc__)) ;







extern int fclose (FILE *__stream);




extern int fflush (FILE *__stream);
# 227 "/usr/include/stdio.h" 3 4
extern int fflush_unlocked (FILE *__stream);
# 237 "/usr/include/stdio.h" 3 4
extern int fcloseall (void);
# 246 "/usr/include/stdio.h" 3 4
extern FILE *fopen (const char *__restrict __filename,
      const char *__restrict __modes) ;




extern FILE *freopen (const char *__restrict __filename,
        const char *__restrict __modes,
        FILE *__restrict __stream) ;
# 270 "/usr/include/stdio.h" 3 4
extern FILE *fopen64 (const char *__restrict __filename,
        const char *__restrict __modes) ;
extern FILE *freopen64 (const char *__restrict __filename,
   const char *__restrict __modes,
   FILE *__restrict __stream) ;




extern FILE *fdopen (int __fd, const char *__modes) throw () ;





extern FILE *fopencookie (void *__restrict __magic_cookie,
     const char *__restrict __modes,
     cookie_io_functions_t __io_funcs) throw () ;




extern FILE *fmemopen (void *__s, size_t __len, const char *__modes)
  throw () ;




extern FILE *open_memstream (char **__bufloc, size_t *__sizeloc) throw () ;





extern void setbuf (FILE *__restrict __stream, char *__restrict __buf) throw ();



extern int setvbuf (FILE *__restrict __stream, char *__restrict __buf,
      int __modes, size_t __n) throw ();




extern void setbuffer (FILE *__restrict __stream, char *__restrict __buf,
         size_t __size) throw ();


extern void setlinebuf (FILE *__stream) throw ();







extern int fprintf (FILE *__restrict __stream,
      const char *__restrict __format, ...);




extern int printf (const char *__restrict __format, ...);

extern int sprintf (char *__restrict __s,
      const char *__restrict __format, ...) throw ();





extern int vfprintf (FILE *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg);




extern int vprintf (const char *__restrict __format, __gnuc_va_list __arg);

extern int vsprintf (char *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg) throw ();



extern int snprintf (char *__restrict __s, size_t __maxlen,
       const char *__restrict __format, ...)
     throw () __attribute__ ((__format__ (__printf__, 3, 4)));

extern int vsnprintf (char *__restrict __s, size_t __maxlen,
        const char *__restrict __format, __gnuc_va_list __arg)
     throw () __attribute__ ((__format__ (__printf__, 3, 0)));





extern int vasprintf (char **__restrict __ptr, const char *__restrict __f,
        __gnuc_va_list __arg)
     throw () __attribute__ ((__format__ (__printf__, 2, 0))) ;
extern int __asprintf (char **__restrict __ptr,
         const char *__restrict __fmt, ...)
     throw () __attribute__ ((__format__ (__printf__, 2, 3))) ;
extern int asprintf (char **__restrict __ptr,
       const char *__restrict __fmt, ...)
     throw () __attribute__ ((__format__ (__printf__, 2, 3))) ;




extern int vdprintf (int __fd, const char *__restrict __fmt,
       __gnuc_va_list __arg)
     __attribute__ ((__format__ (__printf__, 2, 0)));
extern int dprintf (int __fd, const char *__restrict __fmt, ...)
     __attribute__ ((__format__ (__printf__, 2, 3)));







extern int fscanf (FILE *__restrict __stream,
     const char *__restrict __format, ...) ;




extern int scanf (const char *__restrict __format, ...) ;

extern int sscanf (const char *__restrict __s,
     const char *__restrict __format, ...) throw ();





# 1 "/usr/include/bits/floatn.h" 1 3 4
# 74 "/usr/include/bits/floatn.h" 3 4
typedef _Complex float __cfloat128 __attribute__ ((__mode__ (__TC__)));
# 86 "/usr/include/bits/floatn.h" 3 4
typedef __float128 _Float128;
# 119 "/usr/include/bits/floatn.h" 3 4
# 1 "/usr/include/bits/floatn-common.h" 1 3 4
# 24 "/usr/include/bits/floatn-common.h" 3 4
# 1 "/usr/include/bits/long-double.h" 1 3 4
# 25 "/usr/include/bits/floatn-common.h" 2 3 4
# 214 "/usr/include/bits/floatn-common.h" 3 4
typedef float _Float32;
# 251 "/usr/include/bits/floatn-common.h" 3 4
typedef double _Float64;
# 268 "/usr/include/bits/floatn-common.h" 3 4
typedef double _Float32x;
# 285 "/usr/include/bits/floatn-common.h" 3 4
typedef long double _Float64x;
# 120 "/usr/include/bits/floatn.h" 2 3 4
# 407 "/usr/include/stdio.h" 2 3 4



extern int fscanf (FILE *__restrict __stream, const char *__restrict __format, ...) __asm__ ("" "__isoc99_fscanf")

                               ;
extern int scanf (const char *__restrict __format, ...) __asm__ ("" "__isoc99_scanf")
                              ;
extern int sscanf (const char *__restrict __s, const char *__restrict __format, ...) throw () __asm__ ("" "__isoc99_sscanf")

                      ;
# 435 "/usr/include/stdio.h" 3 4
extern int vfscanf (FILE *__restrict __s, const char *__restrict __format,
      __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 2, 0))) ;





extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 1, 0))) ;


extern int vsscanf (const char *__restrict __s,
      const char *__restrict __format, __gnuc_va_list __arg)
     throw () __attribute__ ((__format__ (__scanf__, 2, 0)));





extern int vfscanf (FILE *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vfscanf")



     __attribute__ ((__format__ (__scanf__, 2, 0))) ;
extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vscanf")

     __attribute__ ((__format__ (__scanf__, 1, 0))) ;
extern int vsscanf (const char *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) throw () __asm__ ("" "__isoc99_vsscanf")



     __attribute__ ((__format__ (__scanf__, 2, 0)));
# 489 "/usr/include/stdio.h" 3 4
extern int fgetc (FILE *__stream);
extern int getc (FILE *__stream);





extern int getchar (void);






extern int getc_unlocked (FILE *__stream);
extern int getchar_unlocked (void);
# 514 "/usr/include/stdio.h" 3 4
extern int fgetc_unlocked (FILE *__stream);
# 525 "/usr/include/stdio.h" 3 4
extern int fputc (int __c, FILE *__stream);
extern int putc (int __c, FILE *__stream);





extern int putchar (int __c);
# 541 "/usr/include/stdio.h" 3 4
extern int fputc_unlocked (int __c, FILE *__stream);







extern int putc_unlocked (int __c, FILE *__stream);
extern int putchar_unlocked (int __c);






extern int getw (FILE *__stream);


extern int putw (int __w, FILE *__stream);







extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
     __attribute__ ((__access__ (__write_only__, 1, 2)));
# 591 "/usr/include/stdio.h" 3 4
extern char *fgets_unlocked (char *__restrict __s, int __n,
        FILE *__restrict __stream)
    __attribute__ ((__access__ (__write_only__, 1, 2)));
# 608 "/usr/include/stdio.h" 3 4
extern __ssize_t __getdelim (char **__restrict __lineptr,
                             size_t *__restrict __n, int __delimiter,
                             FILE *__restrict __stream) ;
extern __ssize_t getdelim (char **__restrict __lineptr,
                           size_t *__restrict __n, int __delimiter,
                           FILE *__restrict __stream) ;







extern __ssize_t getline (char **__restrict __lineptr,
                          size_t *__restrict __n,
                          FILE *__restrict __stream) ;







extern int fputs (const char *__restrict __s, FILE *__restrict __stream);





extern int puts (const char *__s);






extern int ungetc (int __c, FILE *__stream);






extern size_t fread (void *__restrict __ptr, size_t __size,
       size_t __n, FILE *__restrict __stream) ;




extern size_t fwrite (const void *__restrict __ptr, size_t __size,
        size_t __n, FILE *__restrict __s);
# 667 "/usr/include/stdio.h" 3 4
extern int fputs_unlocked (const char *__restrict __s,
      FILE *__restrict __stream);
# 678 "/usr/include/stdio.h" 3 4
extern size_t fread_unlocked (void *__restrict __ptr, size_t __size,
         size_t __n, FILE *__restrict __stream) ;
extern size_t fwrite_unlocked (const void *__restrict __ptr, size_t __size,
          size_t __n, FILE *__restrict __stream);







extern int fseek (FILE *__stream, long int __off, int __whence);




extern long int ftell (FILE *__stream) ;




extern void rewind (FILE *__stream);
# 712 "/usr/include/stdio.h" 3 4
extern int fseeko (FILE *__stream, __off_t __off, int __whence);




extern __off_t ftello (FILE *__stream) ;
# 736 "/usr/include/stdio.h" 3 4
extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos);




extern int fsetpos (FILE *__stream, const fpos_t *__pos);
# 755 "/usr/include/stdio.h" 3 4
extern int fseeko64 (FILE *__stream, __off64_t __off, int __whence);
extern __off64_t ftello64 (FILE *__stream) ;
extern int fgetpos64 (FILE *__restrict __stream, fpos64_t *__restrict __pos);
extern int fsetpos64 (FILE *__stream, const fpos64_t *__pos);



extern void clearerr (FILE *__stream) throw ();

extern int feof (FILE *__stream) throw () ;

extern int ferror (FILE *__stream) throw () ;



extern void clearerr_unlocked (FILE *__stream) throw ();
extern int feof_unlocked (FILE *__stream) throw () ;
extern int ferror_unlocked (FILE *__stream) throw () ;







extern void perror (const char *__s);




extern int fileno (FILE *__stream) throw () ;




extern int fileno_unlocked (FILE *__stream) throw () ;
# 799 "/usr/include/stdio.h" 3 4
extern FILE *popen (const char *__command, const char *__modes) ;





extern int pclose (FILE *__stream);





extern char *ctermid (char *__s) throw ();





extern char *cuserid (char *__s);




struct obstack;


extern int obstack_printf (struct obstack *__restrict __obstack,
      const char *__restrict __format, ...)
     throw () __attribute__ ((__format__ (__printf__, 2, 3)));
extern int obstack_vprintf (struct obstack *__restrict __obstack,
       const char *__restrict __format,
       __gnuc_va_list __args)
     throw () __attribute__ ((__format__ (__printf__, 2, 0)));







extern void flockfile (FILE *__stream) throw ();



extern int ftrylockfile (FILE *__stream) throw () ;


extern void funlockfile (FILE *__stream) throw ();
# 857 "/usr/include/stdio.h" 3 4
extern int __uflow (FILE *);
extern int __overflow (FILE *, int);
# 874 "/usr/include/stdio.h" 3 4
}
# 137 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 2
# 1 "/usr/include/c++/10/stdlib.h" 1 3
# 36 "/usr/include/c++/10/stdlib.h" 3
# 1 "/usr/include/c++/10/cstdlib" 1 3
# 39 "/usr/include/c++/10/cstdlib" 3
       
# 40 "/usr/include/c++/10/cstdlib" 3
# 75 "/usr/include/c++/10/cstdlib" 3
# 1 "/usr/include/stdlib.h" 1 3 4
# 25 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 26 "/usr/include/stdlib.h" 2 3 4





# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 1 3 4
# 32 "/usr/include/stdlib.h" 2 3 4

extern "C" {





# 1 "/usr/include/bits/waitflags.h" 1 3 4
# 40 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/bits/waitstatus.h" 1 3 4
# 41 "/usr/include/stdlib.h" 2 3 4
# 58 "/usr/include/stdlib.h" 3 4
typedef struct
  {
    int quot;
    int rem;
  } div_t;



typedef struct
  {
    long int quot;
    long int rem;
  } ldiv_t;





__extension__ typedef struct
  {
    long long int quot;
    long long int rem;
  } lldiv_t;
# 97 "/usr/include/stdlib.h" 3 4
extern size_t __ctype_get_mb_cur_max (void) throw () ;



extern double atof (const char *__nptr)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern int atoi (const char *__nptr)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern long int atol (const char *__nptr)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;



__extension__ extern long long int atoll (const char *__nptr)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;



extern double strtod (const char *__restrict __nptr,
        char **__restrict __endptr)
     throw () __attribute__ ((__nonnull__ (1)));



extern float strtof (const char *__restrict __nptr,
       char **__restrict __endptr) throw () __attribute__ ((__nonnull__ (1)));

extern long double strtold (const char *__restrict __nptr,
       char **__restrict __endptr)
     throw () __attribute__ ((__nonnull__ (1)));
# 140 "/usr/include/stdlib.h" 3 4
extern _Float32 strtof32 (const char *__restrict __nptr,
     char **__restrict __endptr)
     throw () __attribute__ ((__nonnull__ (1)));



extern _Float64 strtof64 (const char *__restrict __nptr,
     char **__restrict __endptr)
     throw () __attribute__ ((__nonnull__ (1)));



extern _Float128 strtof128 (const char *__restrict __nptr,
       char **__restrict __endptr)
     throw () __attribute__ ((__nonnull__ (1)));



extern _Float32x strtof32x (const char *__restrict __nptr,
       char **__restrict __endptr)
     throw () __attribute__ ((__nonnull__ (1)));



extern _Float64x strtof64x (const char *__restrict __nptr,
       char **__restrict __endptr)
     throw () __attribute__ ((__nonnull__ (1)));
# 176 "/usr/include/stdlib.h" 3 4
extern long int strtol (const char *__restrict __nptr,
   char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));

extern unsigned long int strtoul (const char *__restrict __nptr,
      char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));



__extension__
extern long long int strtoq (const char *__restrict __nptr,
        char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));

__extension__
extern unsigned long long int strtouq (const char *__restrict __nptr,
           char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));




__extension__
extern long long int strtoll (const char *__restrict __nptr,
         char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));

__extension__
extern unsigned long long int strtoull (const char *__restrict __nptr,
     char **__restrict __endptr, int __base)
     throw () __attribute__ ((__nonnull__ (1)));




extern int strfromd (char *__dest, size_t __size, const char *__format,
       double __f)
     throw () __attribute__ ((__nonnull__ (3)));

extern int strfromf (char *__dest, size_t __size, const char *__format,
       float __f)
     throw () __attribute__ ((__nonnull__ (3)));

extern int strfroml (char *__dest, size_t __size, const char *__format,
       long double __f)
     throw () __attribute__ ((__nonnull__ (3)));
# 232 "/usr/include/stdlib.h" 3 4
extern int strfromf32 (char *__dest, size_t __size, const char * __format,
         _Float32 __f)
     throw () __attribute__ ((__nonnull__ (3)));



extern int strfromf64 (char *__dest, size_t __size, const char * __format,
         _Float64 __f)
     throw () __attribute__ ((__nonnull__ (3)));



extern int strfromf128 (char *__dest, size_t __size, const char * __format,
   _Float128 __f)
     throw () __attribute__ ((__nonnull__ (3)));



extern int strfromf32x (char *__dest, size_t __size, const char * __format,
   _Float32x __f)
     throw () __attribute__ ((__nonnull__ (3)));



extern int strfromf64x (char *__dest, size_t __size, const char * __format,
   _Float64x __f)
     throw () __attribute__ ((__nonnull__ (3)));
# 274 "/usr/include/stdlib.h" 3 4
extern long int strtol_l (const char *__restrict __nptr,
     char **__restrict __endptr, int __base,
     locale_t __loc) throw () __attribute__ ((__nonnull__ (1, 4)));

extern unsigned long int strtoul_l (const char *__restrict __nptr,
        char **__restrict __endptr,
        int __base, locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 4)));

__extension__
extern long long int strtoll_l (const char *__restrict __nptr,
    char **__restrict __endptr, int __base,
    locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 4)));

__extension__
extern unsigned long long int strtoull_l (const char *__restrict __nptr,
       char **__restrict __endptr,
       int __base, locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 4)));

extern double strtod_l (const char *__restrict __nptr,
   char **__restrict __endptr, locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 3)));

extern float strtof_l (const char *__restrict __nptr,
         char **__restrict __endptr, locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 3)));

extern long double strtold_l (const char *__restrict __nptr,
         char **__restrict __endptr,
         locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 3)));
# 316 "/usr/include/stdlib.h" 3 4
extern _Float32 strtof32_l (const char *__restrict __nptr,
       char **__restrict __endptr,
       locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 3)));



extern _Float64 strtof64_l (const char *__restrict __nptr,
       char **__restrict __endptr,
       locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 3)));



extern _Float128 strtof128_l (const char *__restrict __nptr,
         char **__restrict __endptr,
         locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 3)));



extern _Float32x strtof32x_l (const char *__restrict __nptr,
         char **__restrict __endptr,
         locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 3)));



extern _Float64x strtof64x_l (const char *__restrict __nptr,
         char **__restrict __endptr,
         locale_t __loc)
     throw () __attribute__ ((__nonnull__ (1, 3)));
# 385 "/usr/include/stdlib.h" 3 4
extern char *l64a (long int __n) throw () ;


extern long int a64l (const char *__s)
     throw () __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;




# 1 "/usr/include/sys/types.h" 1 3 4
# 27 "/usr/include/sys/types.h" 3 4
extern "C" {





typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;


typedef __loff_t loff_t;




typedef __ino_t ino_t;






typedef __ino64_t ino64_t;




typedef __dev_t dev_t;




typedef __gid_t gid_t;




typedef __mode_t mode_t;




typedef __nlink_t nlink_t;




typedef __uid_t uid_t;
# 103 "/usr/include/sys/types.h" 3 4
typedef __id_t id_t;
# 114 "/usr/include/sys/types.h" 3 4
typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;





typedef __key_t key_t;
# 134 "/usr/include/sys/types.h" 3 4
typedef __useconds_t useconds_t;



typedef __suseconds_t suseconds_t;





# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 1 3 4
# 145 "/usr/include/sys/types.h" 2 3 4



typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;




# 1 "/usr/include/bits/stdint-intn.h" 1 3 4
# 24 "/usr/include/bits/stdint-intn.h" 3 4
typedef __int8_t int8_t;
typedef __int16_t int16_t;
typedef __int32_t int32_t;
typedef __int64_t int64_t;
# 156 "/usr/include/sys/types.h" 2 3 4


typedef __uint8_t u_int8_t;
typedef __uint16_t u_int16_t;
typedef __uint32_t u_int32_t;
typedef __uint64_t u_int64_t;


typedef int register_t __attribute__ ((__mode__ (__word__)));
# 176 "/usr/include/sys/types.h" 3 4
# 1 "/usr/include/endian.h" 1 3 4
# 35 "/usr/include/endian.h" 3 4
# 1 "/usr/include/bits/byteswap.h" 1 3 4
# 33 "/usr/include/bits/byteswap.h" 3 4
static __inline __uint16_t
__bswap_16 (__uint16_t __bsx)
{

  return __builtin_bswap16 (__bsx);



}






static __inline __uint32_t
__bswap_32 (__uint32_t __bsx)
{

  return __builtin_bswap32 (__bsx);



}
# 69 "/usr/include/bits/byteswap.h" 3 4
__extension__ static __inline __uint64_t
__bswap_64 (__uint64_t __bsx)
{

  return __builtin_bswap64 (__bsx);



}
# 36 "/usr/include/endian.h" 2 3 4
# 1 "/usr/include/bits/uintn-identity.h" 1 3 4
# 32 "/usr/include/bits/uintn-identity.h" 3 4
static __inline __uint16_t
__uint16_identity (__uint16_t __x)
{
  return __x;
}

static __inline __uint32_t
__uint32_identity (__uint32_t __x)
{
  return __x;
}

static __inline __uint64_t
__uint64_identity (__uint64_t __x)
{
  return __x;
}
# 37 "/usr/include/endian.h" 2 3 4
# 177 "/usr/include/sys/types.h" 2 3 4


# 1 "/usr/include/sys/select.h" 1 3 4
# 30 "/usr/include/sys/select.h" 3 4
# 1 "/usr/include/bits/select.h" 1 3 4
# 31 "/usr/include/sys/select.h" 2 3 4


# 1 "/usr/include/bits/types/sigset_t.h" 1 3 4



# 1 "/usr/include/bits/types/__sigset_t.h" 1 3 4




typedef struct
{
  unsigned long int __val[(1024 / (8 * sizeof (unsigned long int)))];
} __sigset_t;
# 5 "/usr/include/bits/types/sigset_t.h" 2 3 4


typedef __sigset_t sigset_t;
# 34 "/usr/include/sys/select.h" 2 3 4
# 49 "/usr/include/sys/select.h" 3 4
typedef long int __fd_mask;
# 59 "/usr/include/sys/select.h" 3 4
typedef struct
  {



    __fd_mask fds_bits[1024 / (8 * (int) sizeof (__fd_mask))];





  } fd_set;






typedef __fd_mask fd_mask;
# 91 "/usr/include/sys/select.h" 3 4
extern "C" {
# 101 "/usr/include/sys/select.h" 3 4
extern int select (int __nfds, fd_set *__restrict __readfds,
     fd_set *__restrict __writefds,
     fd_set *__restrict __exceptfds,
     struct timeval *__restrict __timeout);
# 113 "/usr/include/sys/select.h" 3 4
extern int pselect (int __nfds, fd_set *__restrict __readfds,
      fd_set *__restrict __writefds,
      fd_set *__restrict __exceptfds,
      const struct timespec *__restrict __timeout,
      const __sigset_t *__restrict __sigmask);
# 126 "/usr/include/sys/select.h" 3 4
}
# 180 "/usr/include/sys/types.h" 2 3 4





typedef __blksize_t blksize_t;






typedef __blkcnt_t blkcnt_t;



typedef __fsblkcnt_t fsblkcnt_t;



typedef __fsfilcnt_t fsfilcnt_t;
# 219 "/usr/include/sys/types.h" 3 4
typedef __blkcnt64_t blkcnt64_t;
typedef __fsblkcnt64_t fsblkcnt64_t;
typedef __fsfilcnt64_t fsfilcnt64_t;





# 1 "/usr/include/bits/pthreadtypes.h" 1 3 4
# 23 "/usr/include/bits/pthreadtypes.h" 3 4
# 1 "/usr/include/bits/thread-shared-types.h" 1 3 4
# 44 "/usr/include/bits/thread-shared-types.h" 3 4
# 1 "/usr/include/bits/pthreadtypes-arch.h" 1 3 4
# 21 "/usr/include/bits/pthreadtypes-arch.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 22 "/usr/include/bits/pthreadtypes-arch.h" 2 3 4
# 45 "/usr/include/bits/thread-shared-types.h" 2 3 4




typedef struct __pthread_internal_list
{
  struct __pthread_internal_list *__prev;
  struct __pthread_internal_list *__next;
} __pthread_list_t;

typedef struct __pthread_internal_slist
{
  struct __pthread_internal_slist *__next;
} __pthread_slist_t;
# 74 "/usr/include/bits/thread-shared-types.h" 3 4
# 1 "/usr/include/bits/struct_mutex.h" 1 3 4
# 22 "/usr/include/bits/struct_mutex.h" 3 4
struct __pthread_mutex_s
{
  int __lock;
  unsigned int __count;
  int __owner;

  unsigned int __nusers;



  int __kind;

  short __spins;
  short __elision;
  __pthread_list_t __list;
# 53 "/usr/include/bits/struct_mutex.h" 3 4
};
# 75 "/usr/include/bits/thread-shared-types.h" 2 3 4
# 87 "/usr/include/bits/thread-shared-types.h" 3 4
# 1 "/usr/include/bits/struct_rwlock.h" 1 3 4
# 23 "/usr/include/bits/struct_rwlock.h" 3 4
struct __pthread_rwlock_arch_t
{
  unsigned int __readers;
  unsigned int __writers;
  unsigned int __wrphase_futex;
  unsigned int __writers_futex;
  unsigned int __pad3;
  unsigned int __pad4;

  int __cur_writer;
  int __shared;
  signed char __rwelision;




  unsigned char __pad1[7];


  unsigned long int __pad2;


  unsigned int __flags;
# 55 "/usr/include/bits/struct_rwlock.h" 3 4
};
# 88 "/usr/include/bits/thread-shared-types.h" 2 3 4




struct __pthread_cond_s
{
  __extension__ union
  {
    __extension__ unsigned long long int __wseq;
    struct
    {
      unsigned int __low;
      unsigned int __high;
    } __wseq32;
  };
  __extension__ union
  {
    __extension__ unsigned long long int __g1_start;
    struct
    {
      unsigned int __low;
      unsigned int __high;
    } __g1_start32;
  };
  unsigned int __g_refs[2] ;
  unsigned int __g_size[2];
  unsigned int __g1_orig_size;
  unsigned int __wrefs;
  unsigned int __g_signals[2];
};

typedef unsigned int __tss_t;
typedef unsigned long int __thrd_t;

typedef struct
{
  int __data ;
} __once_flag;
# 24 "/usr/include/bits/pthreadtypes.h" 2 3 4



typedef unsigned long int pthread_t;




typedef union
{
  char __size[4];
  int __align;
} pthread_mutexattr_t;




typedef union
{
  char __size[4];
  int __align;
} pthread_condattr_t;



typedef unsigned int pthread_key_t;



typedef int pthread_once_t;


union pthread_attr_t
{
  char __size[56];
  long int __align;
};

typedef union pthread_attr_t pthread_attr_t;




typedef union
{
  struct __pthread_mutex_s __data;
  char __size[40];
  long int __align;
} pthread_mutex_t;


typedef union
{
  struct __pthread_cond_s __data;
  char __size[48];
  __extension__ long long int __align;
} pthread_cond_t;





typedef union
{
  struct __pthread_rwlock_arch_t __data;
  char __size[56];
  long int __align;
} pthread_rwlock_t;

typedef union
{
  char __size[8];
  long int __align;
} pthread_rwlockattr_t;





typedef volatile int pthread_spinlock_t;




typedef union
{
  char __size[32];
  long int __align;
} pthread_barrier_t;

typedef union
{
  char __size[4];
  int __align;
} pthread_barrierattr_t;
# 228 "/usr/include/sys/types.h" 2 3 4


}
# 395 "/usr/include/stdlib.h" 2 3 4






extern long int random (void) throw ();


extern void srandom (unsigned int __seed) throw ();





extern char *initstate (unsigned int __seed, char *__statebuf,
   size_t __statelen) throw () __attribute__ ((__nonnull__ (2)));



extern char *setstate (char *__statebuf) throw () __attribute__ ((__nonnull__ (1)));







struct random_data
  {
    int32_t *fptr;
    int32_t *rptr;
    int32_t *state;
    int rand_type;
    int rand_deg;
    int rand_sep;
    int32_t *end_ptr;
  };

extern int random_r (struct random_data *__restrict __buf,
       int32_t *__restrict __result) throw () __attribute__ ((__nonnull__ (1, 2)));

extern int srandom_r (unsigned int __seed, struct random_data *__buf)
     throw () __attribute__ ((__nonnull__ (2)));

extern int initstate_r (unsigned int __seed, char *__restrict __statebuf,
   size_t __statelen,
   struct random_data *__restrict __buf)
     throw () __attribute__ ((__nonnull__ (2, 4)));

extern int setstate_r (char *__restrict __statebuf,
         struct random_data *__restrict __buf)
     throw () __attribute__ ((__nonnull__ (1, 2)));





extern int rand (void) throw ();

extern void srand (unsigned int __seed) throw ();



extern int rand_r (unsigned int *__seed) throw ();







extern double drand48 (void) throw ();
extern double erand48 (unsigned short int __xsubi[3]) throw () __attribute__ ((__nonnull__ (1)));


extern long int lrand48 (void) throw ();
extern long int nrand48 (unsigned short int __xsubi[3])
     throw () __attribute__ ((__nonnull__ (1)));


extern long int mrand48 (void) throw ();
extern long int jrand48 (unsigned short int __xsubi[3])
     throw () __attribute__ ((__nonnull__ (1)));


extern void srand48 (long int __seedval) throw ();
extern unsigned short int *seed48 (unsigned short int __seed16v[3])
     throw () __attribute__ ((__nonnull__ (1)));
extern void lcong48 (unsigned short int __param[7]) throw () __attribute__ ((__nonnull__ (1)));





struct drand48_data
  {
    unsigned short int __x[3];
    unsigned short int __old_x[3];
    unsigned short int __c;
    unsigned short int __init;
    __extension__ unsigned long long int __a;

  };


extern int drand48_r (struct drand48_data *__restrict __buffer,
        double *__restrict __result) throw () __attribute__ ((__nonnull__ (1, 2)));
extern int erand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        double *__restrict __result) throw () __attribute__ ((__nonnull__ (1, 2)));


extern int lrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     throw () __attribute__ ((__nonnull__ (1, 2)));
extern int nrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     throw () __attribute__ ((__nonnull__ (1, 2)));


extern int mrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     throw () __attribute__ ((__nonnull__ (1, 2)));
extern int jrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     throw () __attribute__ ((__nonnull__ (1, 2)));


extern int srand48_r (long int __seedval, struct drand48_data *__buffer)
     throw () __attribute__ ((__nonnull__ (2)));

extern int seed48_r (unsigned short int __seed16v[3],
       struct drand48_data *__buffer) throw () __attribute__ ((__nonnull__ (1, 2)));

extern int lcong48_r (unsigned short int __param[7],
        struct drand48_data *__buffer)
     throw () __attribute__ ((__nonnull__ (1, 2)));




extern void *malloc (size_t __size) throw () __attribute__ ((__malloc__))
     __attribute__ ((__alloc_size__ (1))) ;

extern void *calloc (size_t __nmemb, size_t __size)
     throw () __attribute__ ((__malloc__)) __attribute__ ((__alloc_size__ (1, 2))) ;






extern void *realloc (void *__ptr, size_t __size)
     throw () __attribute__ ((__warn_unused_result__)) __attribute__ ((__alloc_size__ (2)));







extern void *reallocarray (void *__ptr, size_t __nmemb, size_t __size)
     throw () __attribute__ ((__warn_unused_result__))
     __attribute__ ((__alloc_size__ (2, 3)));



extern void free (void *__ptr) throw ();


# 1 "/usr/include/alloca.h" 1 3 4
# 24 "/usr/include/alloca.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stddef.h" 1 3 4
# 25 "/usr/include/alloca.h" 2 3 4

extern "C" {





extern void *alloca (size_t __size) throw ();





}
# 569 "/usr/include/stdlib.h" 2 3 4





extern void *valloc (size_t __size) throw () __attribute__ ((__malloc__))
     __attribute__ ((__alloc_size__ (1))) ;




extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size)
     throw () __attribute__ ((__nonnull__ (1))) ;




extern void *aligned_alloc (size_t __alignment, size_t __size)
     throw () __attribute__ ((__malloc__)) __attribute__ ((__alloc_size__ (2))) ;



extern void abort (void) throw () __attribute__ ((__noreturn__));



extern int atexit (void (*__func) (void)) throw () __attribute__ ((__nonnull__ (1)));




extern "C++" int at_quick_exit (void (*__func) (void))
     throw () __asm ("at_quick_exit") __attribute__ ((__nonnull__ (1)));
# 610 "/usr/include/stdlib.h" 3 4
extern int on_exit (void (*__func) (int __status, void *__arg), void *__arg)
     throw () __attribute__ ((__nonnull__ (1)));





extern void exit (int __status) throw () __attribute__ ((__noreturn__));





extern void quick_exit (int __status) throw () __attribute__ ((__noreturn__));





extern void _Exit (int __status) throw () __attribute__ ((__noreturn__));




extern char *getenv (const char *__name) throw () __attribute__ ((__nonnull__ (1))) ;




extern char *secure_getenv (const char *__name)
     throw () __attribute__ ((__nonnull__ (1))) ;






extern int putenv (char *__string) throw () __attribute__ ((__nonnull__ (1)));





extern int setenv (const char *__name, const char *__value, int __replace)
     throw () __attribute__ ((__nonnull__ (2)));


extern int unsetenv (const char *__name) throw () __attribute__ ((__nonnull__ (1)));






extern int clearenv (void) throw ();
# 675 "/usr/include/stdlib.h" 3 4
extern char *mktemp (char *__template) throw () __attribute__ ((__nonnull__ (1)));
# 688 "/usr/include/stdlib.h" 3 4
extern int mkstemp (char *__template) __attribute__ ((__nonnull__ (1))) ;
# 698 "/usr/include/stdlib.h" 3 4
extern int mkstemp64 (char *__template) __attribute__ ((__nonnull__ (1))) ;
# 710 "/usr/include/stdlib.h" 3 4
extern int mkstemps (char *__template, int __suffixlen) __attribute__ ((__nonnull__ (1))) ;
# 720 "/usr/include/stdlib.h" 3 4
extern int mkstemps64 (char *__template, int __suffixlen)
     __attribute__ ((__nonnull__ (1))) ;
# 731 "/usr/include/stdlib.h" 3 4
extern char *mkdtemp (char *__template) throw () __attribute__ ((__nonnull__ (1))) ;
# 742 "/usr/include/stdlib.h" 3 4
extern int mkostemp (char *__template, int __flags) __attribute__ ((__nonnull__ (1))) ;
# 752 "/usr/include/stdlib.h" 3 4
extern int mkostemp64 (char *__template, int __flags) __attribute__ ((__nonnull__ (1))) ;
# 762 "/usr/include/stdlib.h" 3 4
extern int mkostemps (char *__template, int __suffixlen, int __flags)
     __attribute__ ((__nonnull__ (1))) ;
# 774 "/usr/include/stdlib.h" 3 4
extern int mkostemps64 (char *__template, int __suffixlen, int __flags)
     __attribute__ ((__nonnull__ (1))) ;
# 784 "/usr/include/stdlib.h" 3 4
extern int system (const char *__command) ;





extern char *canonicalize_file_name (const char *__name)
     throw () __attribute__ ((__nonnull__ (1))) ;
# 800 "/usr/include/stdlib.h" 3 4
extern char *realpath (const char *__restrict __name,
         char *__restrict __resolved) throw () ;






typedef int (*__compar_fn_t) (const void *, const void *);


typedef __compar_fn_t comparison_fn_t;



typedef int (*__compar_d_fn_t) (const void *, const void *, void *);




extern void *bsearch (const void *__key, const void *__base,
        size_t __nmemb, size_t __size, __compar_fn_t __compar)
     __attribute__ ((__nonnull__ (1, 2, 5))) ;







extern void qsort (void *__base, size_t __nmemb, size_t __size,
     __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 4)));

extern void qsort_r (void *__base, size_t __nmemb, size_t __size,
       __compar_d_fn_t __compar, void *__arg)
  __attribute__ ((__nonnull__ (1, 4)));




extern int abs (int __x) throw () __attribute__ ((__const__)) ;
extern long int labs (long int __x) throw () __attribute__ ((__const__)) ;


__extension__ extern long long int llabs (long long int __x)
     throw () __attribute__ ((__const__)) ;






extern div_t div (int __numer, int __denom)
     throw () __attribute__ ((__const__)) ;
extern ldiv_t ldiv (long int __numer, long int __denom)
     throw () __attribute__ ((__const__)) ;


__extension__ extern lldiv_t lldiv (long long int __numer,
        long long int __denom)
     throw () __attribute__ ((__const__)) ;
# 872 "/usr/include/stdlib.h" 3 4
extern char *ecvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) throw () __attribute__ ((__nonnull__ (3, 4))) ;




extern char *fcvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) throw () __attribute__ ((__nonnull__ (3, 4))) ;




extern char *gcvt (double __value, int __ndigit, char *__buf)
     throw () __attribute__ ((__nonnull__ (3))) ;




extern char *qecvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     throw () __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qfcvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     throw () __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qgcvt (long double __value, int __ndigit, char *__buf)
     throw () __attribute__ ((__nonnull__ (3))) ;




extern int ecvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) throw () __attribute__ ((__nonnull__ (3, 4, 5)));
extern int fcvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) throw () __attribute__ ((__nonnull__ (3, 4, 5)));

extern int qecvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     throw () __attribute__ ((__nonnull__ (3, 4, 5)));
extern int qfcvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     throw () __attribute__ ((__nonnull__ (3, 4, 5)));





extern int mblen (const char *__s, size_t __n) throw ();


extern int mbtowc (wchar_t *__restrict __pwc,
     const char *__restrict __s, size_t __n) throw ();


extern int wctomb (char *__s, wchar_t __wchar) throw ();



extern size_t mbstowcs (wchar_t *__restrict __pwcs,
   const char *__restrict __s, size_t __n) throw ()
    __attribute__ ((__access__ (__read_only__, 2)));

extern size_t wcstombs (char *__restrict __s,
   const wchar_t *__restrict __pwcs, size_t __n)
     throw ()
  __attribute__ ((__access__ (__write_only__, 1, 3))) __attribute__ ((__access__ (__read_only__, 2)));






extern int rpmatch (const char *__response) throw () __attribute__ ((__nonnull__ (1))) ;
# 958 "/usr/include/stdlib.h" 3 4
extern int getsubopt (char **__restrict __optionp,
        char *const *__restrict __tokens,
        char **__restrict __valuep)
     throw () __attribute__ ((__nonnull__ (1, 2, 3))) ;







extern int posix_openpt (int __oflag) ;







extern int grantpt (int __fd) throw ();



extern int unlockpt (int __fd) throw ();




extern char *ptsname (int __fd) throw () ;






extern int ptsname_r (int __fd, char *__buf, size_t __buflen)
     throw () __attribute__ ((__nonnull__ (2))) __attribute__ ((__access__ (__write_only__, 2, 3)));


extern int getpt (void);






extern int getloadavg (double __loadavg[], int __nelem)
     throw () __attribute__ ((__nonnull__ (1)));
# 1014 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/bits/stdlib-float.h" 1 3 4
# 1015 "/usr/include/stdlib.h" 2 3 4
# 1026 "/usr/include/stdlib.h" 3 4
}
# 76 "/usr/include/c++/10/cstdlib" 2 3

# 1 "/usr/include/c++/10/bits/std_abs.h" 1 3
# 33 "/usr/include/c++/10/bits/std_abs.h" 3
       
# 34 "/usr/include/c++/10/bits/std_abs.h" 3
# 46 "/usr/include/c++/10/bits/std_abs.h" 3
extern "C++"
{
namespace std __attribute__ ((__visibility__ ("default")))
{


  using ::abs;


  inline long
  abs(long __i) { return __builtin_labs(__i); }



  inline long long
  abs(long long __x) { return __builtin_llabs (__x); }
# 70 "/usr/include/c++/10/bits/std_abs.h" 3
  inline constexpr double
  abs(double __x)
  { return __builtin_fabs(__x); }

  inline constexpr float
  abs(float __x)
  { return __builtin_fabsf(__x); }

  inline constexpr long double
  abs(long double __x)
  { return __builtin_fabsl(__x); }



  inline constexpr __int128
  abs(__int128 __x) { return __x >= 0 ? __x : -__x; }
# 101 "/usr/include/c++/10/bits/std_abs.h" 3
  inline constexpr
  __float128
  abs(__float128 __x)
  { return __x < 0 ? -__x : __x; }



}
}
# 78 "/usr/include/c++/10/cstdlib" 2 3
# 121 "/usr/include/c++/10/cstdlib" 3
extern "C++"
{
namespace std __attribute__ ((__visibility__ ("default")))
{


  using ::div_t;
  using ::ldiv_t;

  using ::abort;



  using ::atexit;


  using ::at_quick_exit;


  using ::atof;
  using ::atoi;
  using ::atol;
  using ::bsearch;
  using ::calloc;
  using ::div;
  using ::exit;
  using ::free;
  using ::getenv;
  using ::labs;
  using ::ldiv;
  using ::malloc;

  using ::mblen;
  using ::mbstowcs;
  using ::mbtowc;

  using ::qsort;


  using ::quick_exit;


  using ::rand;
  using ::realloc;
  using ::srand;
  using ::strtod;
  using ::strtol;
  using ::strtoul;
  using ::system;

  using ::wcstombs;
  using ::wctomb;



  inline ldiv_t
  div(long __i, long __j) { return ldiv(__i, __j); }




}
# 195 "/usr/include/c++/10/cstdlib" 3
namespace __gnu_cxx __attribute__ ((__visibility__ ("default")))
{



  using ::lldiv_t;





  using ::_Exit;



  using ::llabs;

  inline lldiv_t
  div(long long __n, long long __d)
  { lldiv_t __q; __q.quot = __n / __d; __q.rem = __n % __d; return __q; }

  using ::lldiv;
# 227 "/usr/include/c++/10/cstdlib" 3
  using ::atoll;
  using ::strtoll;
  using ::strtoull;

  using ::strtof;
  using ::strtold;


}

namespace std
{

  using ::__gnu_cxx::lldiv_t;

  using ::__gnu_cxx::_Exit;

  using ::__gnu_cxx::llabs;
  using ::__gnu_cxx::div;
  using ::__gnu_cxx::lldiv;

  using ::__gnu_cxx::atoll;
  using ::__gnu_cxx::strtof;
  using ::__gnu_cxx::strtoll;
  using ::__gnu_cxx::strtoull;
  using ::__gnu_cxx::strtold;
}



}
# 37 "/usr/include/c++/10/stdlib.h" 2 3

using std::abort;
using std::atexit;
using std::exit;


  using std::at_quick_exit;


  using std::quick_exit;




using std::div_t;
using std::ldiv_t;

using std::abs;
using std::atof;
using std::atoi;
using std::atol;
using std::bsearch;
using std::calloc;
using std::div;
using std::free;
using std::getenv;
using std::labs;
using std::ldiv;
using std::malloc;

using std::mblen;
using std::mbstowcs;
using std::mbtowc;

using std::qsort;
using std::rand;
using std::realloc;
using std::srand;
using std::strtod;
using std::strtol;
using std::strtoul;
using std::system;

using std::wcstombs;
using std::wctomb;
# 138 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 2






# 143 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern "C"
{
extern







__attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) int printf(const char*, ...);



extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void* malloc(size_t) 
# 157 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 3 4
                                                                                    throw ()
# 157 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
                                                                                           ;
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void free(void*) 
# 158 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 3 4
                                                                                 throw ()
# 158 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
                                                                                        ;

}





# 1 "/usr/include/assert.h" 1 3 4
# 64 "/usr/include/assert.h" 3 4

# 64 "/usr/include/assert.h" 3 4
extern "C" {


extern void __assert_fail (const char *__assertion, const char *__file,
      unsigned int __line, const char *__function)
     throw () __attribute__ ((__noreturn__));


extern void __assert_perror_fail (int __errnum, const char *__file,
      unsigned int __line, const char *__function)
     throw () __attribute__ ((__noreturn__));




extern void __assert (const char *__assertion, const char *__file, int __line)
     throw () __attribute__ ((__noreturn__));


}
# 167 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 2



# 169 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern "C"
{
# 197 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void __assert_fail(
  const char *, const char *, unsigned int, const char *)
  
# 199 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 3 4
 throw ()
# 199 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
        ;




}
# 259 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void* operator new(std:: size_t) ;
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void* operator new[](std:: size_t) ;
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete(void*) throw();
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete[](void*) throw();

extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete(void*, std:: size_t) throw();
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void operator delete[](void*, std:: size_t) throw();
# 295 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 1
# 106 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 1
# 107 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 108 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 2







extern "C"
{
# 213 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) int abs(int a) 
# 213 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                        throw ()
# 213 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                               ;







extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) long int labs(long int a) 
# 221 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                              throw ()
# 221 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                     ;







extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) long long int llabs(long long int a) 
# 229 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                                    throw ()
# 229 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                           ;
# 279 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fabs(double x) 
# 279 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 279 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 320 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fabsf(float x) 
# 320 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 320 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 330 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int min(int a, int b);






extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) unsigned int umin(unsigned int a, unsigned int b);






extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llmin(long long int a, long long int b);






extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) unsigned long long int ullmin(unsigned long long int a, unsigned long long int b);
# 372 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fminf(float x, float y) 
# 372 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 372 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;
# 392 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fmin(double x, double y) 
# 392 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                   throw ()
# 392 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                          ;
# 405 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int max(int a, int b);







extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) unsigned int umax(unsigned int a, unsigned int b);






extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llmax(long long int a, long long int b);






extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) unsigned long long int ullmax(unsigned long long int a, unsigned long long int b);
# 448 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fmaxf(float x, float y) 
# 448 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 448 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;
# 468 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fmax(double, double) 
# 468 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                               throw ()
# 468 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                      ;
# 512 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double sin(double x) 
# 512 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 512 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 545 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double cos(double x) 
# 545 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 545 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 564 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) void sincos(double x, double *sptr, double *cptr) 
# 564 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                                       throw ()
# 564 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                              ;
# 580 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) void sincosf(float x, float *sptr, float *cptr) 
# 580 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                                     throw ()
# 580 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                            ;
# 625 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double tan(double x) 
# 625 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 625 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 694 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double sqrt(double x) 
# 694 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 694 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 766 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double rsqrt(double x);
# 836 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float rsqrtf(float x);
# 892 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double log2(double x) 
# 892 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 892 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 917 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double exp2(double x) 
# 917 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 917 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 942 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float exp2f(float x) 
# 942 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 942 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 969 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double exp10(double x) 
# 969 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 969 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 992 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float exp10f(float x) 
# 992 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 992 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1038 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double expm1(double x) 
# 1038 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 1038 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1083 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float expm1f(float x) 
# 1083 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 1083 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1138 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float log2f(float x) 
# 1138 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 1138 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 1192 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double log10(double x) 
# 1192 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 1192 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1263 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double log(double x) 
# 1263 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 1263 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 1366 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double log1p(double x) 
# 1366 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 1366 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1472 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float log1pf(float x) 
# 1472 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 1472 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1536 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double floor(double x) 
# 1536 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                     throw ()
# 1536 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                            ;
# 1575 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double exp(double x) 
# 1575 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 1575 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 1606 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double cosh(double x) 
# 1606 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 1606 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 1656 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double sinh(double x) 
# 1656 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 1656 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 1686 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double tanh(double x) 
# 1686 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 1686 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 1721 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double acosh(double x) 
# 1721 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 1721 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1759 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float acoshf(float x) 
# 1759 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 1759 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1775 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double asinh(double x) 
# 1775 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 1775 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1791 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float asinhf(float x) 
# 1791 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 1791 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1845 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double atanh(double x) 
# 1845 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 1845 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1899 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float atanhf(float x) 
# 1899 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 1899 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 1958 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double ldexp(double x, int exp) 
# 1958 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                              throw ()
# 1958 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                     ;
# 2014 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float ldexpf(float x, int exp) 
# 2014 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                   throw ()
# 2014 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                          ;
# 2066 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double logb(double x) 
# 2066 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 2066 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 2121 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float logbf(float x) 
# 2121 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 2121 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 2152 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int ilogb(double x) 
# 2152 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 2152 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 2183 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int ilogbf(float x) 
# 2183 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 2183 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 2259 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double scalbn(double x, int n) 
# 2259 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 2259 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;
# 2335 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float scalbnf(float x, int n) 
# 2335 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 2335 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;
# 2411 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double scalbln(double x, long int n) 
# 2411 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                        throw ()
# 2411 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                               ;
# 2487 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float scalblnf(float x, long int n) 
# 2487 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                        throw ()
# 2487 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                               ;
# 2565 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double frexp(double x, int *nptr) 
# 2565 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                                throw ()
# 2565 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                       ;
# 2640 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float frexpf(float x, int *nptr) 
# 2640 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                     throw ()
# 2640 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                            ;
# 2654 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double round(double x) 
# 2654 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 2654 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 2671 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float roundf(float x) 
# 2671 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 2671 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 2689 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long int lround(double x) 
# 2689 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                           throw ()
# 2689 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                  ;
# 2707 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long int lroundf(float x) 
# 2707 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                           throw ()
# 2707 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                  ;
# 2725 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llround(double x) 
# 2725 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                            throw ()
# 2725 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                   ;
# 2743 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llroundf(float x) 
# 2743 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                            throw ()
# 2743 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                   ;
# 2779 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double rint(double x) 
# 2779 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 2779 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 2795 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float rintf(float x) 
# 2795 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 2795 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 2812 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long int lrint(double x) 
# 2812 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 2812 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 2829 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long int lrintf(float x) 
# 2829 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 2829 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 2846 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llrint(double x) 
# 2846 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                           throw ()
# 2846 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                  ;
# 2863 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) long long int llrintf(float x) 
# 2863 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                           throw ()
# 2863 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                  ;
# 2916 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double nearbyint(double x) 
# 2916 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                              throw ()
# 2916 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                     ;
# 2969 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float nearbyintf(float x) 
# 2969 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                              throw ()
# 2969 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                     ;
# 3031 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double ceil(double x) 
# 3031 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                    throw ()
# 3031 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                           ;
# 3043 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double trunc(double x) 
# 3043 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 3043 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 3058 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float truncf(float x) 
# 3058 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 3058 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 3084 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fdim(double x, double y) 
# 3084 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                   throw ()
# 3084 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                          ;
# 3110 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fdimf(float x, float y) 
# 3110 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 3110 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;
# 3146 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double atan2(double y, double x) 
# 3146 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                    throw ()
# 3146 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                           ;
# 3177 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double atan(double x) 
# 3177 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 3177 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 3200 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double acos(double x) 
# 3200 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 3200 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 3232 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double asin(double x) 
# 3232 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 3232 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 3278 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double hypot(double x, double y) 
# 3278 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                              throw ()
# 3278 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                     ;
# 3330 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double rhypot(double x, double y) 
# 3330 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                     throw ()
# 3330 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                            ;
# 3376 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float hypotf(float x, float y) 
# 3376 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                            throw ()
# 3376 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                   ;
# 3428 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float rhypotf(float x, float y) 
# 3428 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                    throw ()
# 3428 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                           ;
# 3472 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double norm3d(double a, double b, double c) 
# 3472 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                           throw ()
# 3472 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                  ;
# 3523 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double rnorm3d(double a, double b, double c) 
# 3523 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                throw ()
# 3523 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                       ;
# 3572 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double norm4d(double a, double b, double c, double d) 
# 3572 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                                     throw ()
# 3572 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                            ;
# 3628 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double rnorm4d(double a, double b, double c, double d) 
# 3628 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                           throw ()
# 3628 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                  ;
# 3673 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double norm(int dim, double const * t) 
# 3673 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                           throw ()
# 3673 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                  ;
# 3724 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double rnorm(int dim, double const * t) 
# 3724 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                            throw ()
# 3724 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                   ;
# 3776 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float rnormf(int dim, float const * a) 
# 3776 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                           throw ()
# 3776 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                  ;
# 3820 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float normf(int dim, float const * a) 
# 3820 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                          throw ()
# 3820 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                 ;
# 3865 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float norm3df(float a, float b, float c) 
# 3865 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                             throw ()
# 3865 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                    ;
# 3916 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float rnorm3df(float a, float b, float c) 
# 3916 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                              throw ()
# 3916 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                     ;
# 3965 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float norm4df(float a, float b, float c, float d) 
# 3965 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                      throw ()
# 3965 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                             ;
# 4021 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float rnorm4df(float a, float b, float c, float d) 
# 4021 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                       throw ()
# 4021 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                              ;
# 4108 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double cbrt(double x) 
# 4108 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 4108 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 4194 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float cbrtf(float x) 
# 4194 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 4194 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 4249 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double rcbrt(double x);
# 4299 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float rcbrtf(float x);
# 4359 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double sinpi(double x);
# 4419 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float sinpif(float x);
# 4471 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double cospi(double x);
# 4523 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float cospif(float x);
# 4553 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) void sincospi(double x, double *sptr, double *cptr);
# 4583 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) void sincospif(float x, float *sptr, float *cptr);
# 4895 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double pow(double x, double y) 
# 4895 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 4895 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;
# 4951 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double modf(double x, double *iptr) 
# 4951 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                                  throw ()
# 4951 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                         ;
# 5010 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fmod(double x, double y) 
# 5010 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                   throw ()
# 5010 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                          ;
# 5096 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double remainder(double x, double y) 
# 5096 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                        throw ()
# 5096 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                               ;
# 5186 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float remainderf(float x, float y) 
# 5186 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                       throw ()
# 5186 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                              ;
# 5240 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double remquo(double x, double y, int *quo) 
# 5240 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                               throw ()
# 5240 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                      ;
# 5294 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float remquof(float x, float y, int *quo) 
# 5294 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                              throw ()
# 5294 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                     ;
# 5335 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double j0(double x) 
# 5335 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 5335 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;
# 5377 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float j0f(float x) 
# 5377 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                       throw ()
# 5377 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                              ;
# 5446 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double j1(double x) 
# 5446 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 5446 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;
# 5515 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float j1f(float x) 
# 5515 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                       throw ()
# 5515 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                              ;
# 5558 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double jn(int n, double x) 
# 5558 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                         throw ()
# 5558 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                ;
# 5601 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float jnf(int n, float x) 
# 5601 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                              throw ()
# 5601 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                     ;
# 5653 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double y0(double x) 
# 5653 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 5653 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;
# 5705 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float y0f(float x) 
# 5705 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                       throw ()
# 5705 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                              ;
# 5757 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double y1(double x) 
# 5757 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 5757 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;
# 5809 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float y1f(float x) 
# 5809 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                       throw ()
# 5809 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                              ;
# 5862 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double yn(int n, double x) 
# 5862 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                         throw ()
# 5862 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                ;
# 5915 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float ynf(int n, float x) 
# 5915 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                              throw ()
# 5915 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                     ;
# 5942 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double cyl_bessel_i0(double x) 
# 5942 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                              throw ()
# 5942 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                     ;
# 5968 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float cyl_bessel_i0f(float x) 
# 5968 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                   throw ()
# 5968 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                          ;
# 5995 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double cyl_bessel_i1(double x) 
# 5995 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                              throw ()
# 5995 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                     ;
# 6021 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float cyl_bessel_i1f(float x) 
# 6021 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                   throw ()
# 6021 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                          ;
# 6104 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double erf(double x) 
# 6104 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 6104 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 6186 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float erff(float x) 
# 6186 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 6186 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 6250 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double erfinv(double y);
# 6307 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float erfinvf(float y);
# 6346 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double erfc(double x) 
# 6346 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 6346 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 6384 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float erfcf(float x) 
# 6384 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 6384 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 6512 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double lgamma(double x) 
# 6512 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                           throw ()
# 6512 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                  ;
# 6575 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double erfcinv(double y);
# 6631 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float erfcinvf(float y);
# 6689 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double normcdfinv(double y);
# 6747 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float normcdfinvf(float y);
# 6790 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double normcdf(double y);
# 6833 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float normcdff(float y);
# 6908 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double erfcx(double x);
# 6983 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float erfcxf(float x);
# 7117 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float lgammaf(float x) 
# 7117 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                           throw ()
# 7117 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                  ;
# 7226 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double tgamma(double x) 
# 7226 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                           throw ()
# 7226 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                  ;
# 7335 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float tgammaf(float x) 
# 7335 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                           throw ()
# 7335 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                  ;
# 7348 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double copysign(double x, double y) 
# 7348 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                       throw ()
# 7348 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                              ;
# 7361 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float copysignf(float x, float y) 
# 7361 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                      throw ()
# 7361 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                             ;
# 7380 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double nextafter(double x, double y) 
# 7380 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                        throw ()
# 7380 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                               ;
# 7399 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float nextafterf(float x, float y) 
# 7399 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                       throw ()
# 7399 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                              ;
# 7415 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double nan(const char *tagp) 
# 7415 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                throw ()
# 7415 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                       ;
# 7431 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float nanf(const char *tagp) 
# 7431 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                 throw ()
# 7431 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                        ;






extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isinff(float) 
# 7438 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 7438 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isnanf(float) 
# 7439 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 7439 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 7449 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __finite(double) 
# 7449 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                           throw ()
# 7449 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                  ;
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __finitef(float) 
# 7450 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                           throw ()
# 7450 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                  ;
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __signbit(double) 
# 7451 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                            throw ()
# 7451 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                   ;
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isnan(double) 
# 7452 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 7452 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isinf(double) 
# 7453 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 7453 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;


extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __signbitf(float) 
# 7456 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                            throw ()
# 7456 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                   ;
# 7615 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double fma(double x, double y, double z) 
# 7615 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                            throw ()
# 7615 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                   ;
# 7773 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fmaf(float x, float y, float z) 
# 7773 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                          throw ()
# 7773 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                                 ;
# 7784 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __signbitl(long double) 
# 7784 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 7784 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;





extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __finitel(long double) 
# 7790 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                 throw ()
# 7790 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                        ;
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isinfl(long double) 
# 7791 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                throw ()
# 7791 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                       ;
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) int __isnanl(long double) 
# 7792 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                throw ()
# 7792 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                       ;
# 7842 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float acosf(float x) 
# 7842 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 7842 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 7882 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float asinf(float x) 
# 7882 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 7882 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 7922 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float atanf(float x) 
# 7922 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 7922 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 7955 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float atan2f(float y, float x) 
# 7955 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                   throw ()
# 7955 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                          ;
# 7979 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float cosf(float x) 
# 7979 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 7979 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 8021 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float sinf(float x) 
# 8021 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 8021 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 8063 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float tanf(float x) 
# 8063 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 8063 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 8094 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float coshf(float x) 
# 8094 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 8094 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 8144 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float sinhf(float x) 
# 8144 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 8144 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 8174 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float tanhf(float x) 
# 8174 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 8174 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 8225 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float logf(float x) 
# 8225 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 8225 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 8275 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float expf(float x) 
# 8275 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                        throw ()
# 8275 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                               ;
# 8326 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float log10f(float x) 
# 8326 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 8326 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 8381 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float modff(float x, float *iptr) 
# 8381 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                      throw ()
# 8381 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                             ;
# 8689 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float powf(float x, float y) 
# 8689 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                 throw ()
# 8689 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                        ;
# 8758 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float sqrtf(float x) 
# 8758 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 8758 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 8817 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float ceilf(float x) 
# 8817 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                         throw ()
# 8817 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                ;
# 8878 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float floorf(float x) 
# 8878 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                          throw ()
# 8878 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                 ;
# 8936 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fmodf(float x, float y) 
# 8936 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 3 4
                                                                                                  throw ()
# 8936 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
                                                                                                         ;
# 8951 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}


# 1 "/usr/include/c++/10/math.h" 1 3
# 36 "/usr/include/c++/10/math.h" 3
# 1 "/usr/include/c++/10/cmath" 1 3
# 39 "/usr/include/c++/10/cmath" 3
       
# 40 "/usr/include/c++/10/cmath" 3


# 1 "/usr/include/c++/10/bits/cpp_type_traits.h" 1 3
# 35 "/usr/include/c++/10/bits/cpp_type_traits.h" 3
       
# 36 "/usr/include/c++/10/bits/cpp_type_traits.h" 3
# 67 "/usr/include/c++/10/bits/cpp_type_traits.h" 3

# 67 "/usr/include/c++/10/bits/cpp_type_traits.h" 3
extern "C++" {

namespace std __attribute__ ((__visibility__ ("default")))
{


  struct __true_type { };
  struct __false_type { };

  template<bool>
    struct __truth_type
    { typedef __false_type __type; };

  template<>
    struct __truth_type<true>
    { typedef __true_type __type; };



  template<class _Sp, class _Tp>
    struct __traitor
    {
      enum { __value = bool(_Sp::__value) || bool(_Tp::__value) };
      typedef typename __truth_type<__value>::__type __type;
    };


  template<typename, typename>
    struct __are_same
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<typename _Tp>
    struct __are_same<_Tp, _Tp>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };


  template<typename _Tp>
    struct __is_void
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<>
    struct __is_void<void>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };




  template<typename _Tp>
    struct __is_integer
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };





  template<>
    struct __is_integer<bool>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<signed char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<unsigned char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };


  template<>
    struct __is_integer<wchar_t>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };
# 184 "/usr/include/c++/10/bits/cpp_type_traits.h" 3
  template<>
    struct __is_integer<char16_t>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<char32_t>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };


  template<>
    struct __is_integer<short>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<unsigned short>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<int>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<unsigned int>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<long>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<unsigned long>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<long long>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_integer<unsigned long long>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };
# 270 "/usr/include/c++/10/bits/cpp_type_traits.h" 3
template<> struct __is_integer<__int128> { enum { __value = 1 }; typedef __true_type __type; }; template<> struct __is_integer<unsigned __int128> { enum { __value = 1 }; typedef __true_type __type; };
# 287 "/usr/include/c++/10/bits/cpp_type_traits.h" 3
  template<typename _Tp>
    struct __is_floating
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };


  template<>
    struct __is_floating<float>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_floating<double>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_floating<long double>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };




  template<typename _Tp>
    struct __is_pointer
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<typename _Tp>
    struct __is_pointer<_Tp*>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };




  template<typename _Tp>
    struct __is_arithmetic
    : public __traitor<__is_integer<_Tp>, __is_floating<_Tp> >
    { };




  template<typename _Tp>
    struct __is_scalar
    : public __traitor<__is_arithmetic<_Tp>, __is_pointer<_Tp> >
    { };




  template<typename _Tp>
    struct __is_char
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<>
    struct __is_char<char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };


  template<>
    struct __is_char<wchar_t>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };


  template<typename _Tp>
    struct __is_byte
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };

  template<>
    struct __is_byte<char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_byte<signed char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };

  template<>
    struct __is_byte<unsigned char>
    {
      enum { __value = 1 };
      typedef __true_type __type;
    };
# 423 "/usr/include/c++/10/bits/cpp_type_traits.h" 3
  template<typename> struct iterator_traits;


  template<typename _Tp>
    struct __is_nonvolatile_trivially_copyable
    {
      enum { __value = __is_trivially_copyable(_Tp) };
    };




  template<typename _Tp>
    struct __is_nonvolatile_trivially_copyable<volatile _Tp>
    {
      enum { __value = 0 };
    };


  template<typename _OutputIter, typename _InputIter>
    struct __memcpyable
    {
      enum { __value = 0 };
    };

  template<typename _Tp>
    struct __memcpyable<_Tp*, _Tp*>
    : __is_nonvolatile_trivially_copyable<_Tp>
    { };

  template<typename _Tp>
    struct __memcpyable<_Tp*, const _Tp*>
    : __is_nonvolatile_trivially_copyable<_Tp>
    { };






  template<typename _Iter1, typename _Iter2>
    struct __memcmpable
    {
      enum { __value = 0 };
    };


  template<typename _Tp>
    struct __memcmpable<_Tp*, _Tp*>
    : __is_nonvolatile_trivially_copyable<_Tp>
    { };

  template<typename _Tp>
    struct __memcmpable<const _Tp*, _Tp*>
    : __is_nonvolatile_trivially_copyable<_Tp>
    { };

  template<typename _Tp>
    struct __memcmpable<_Tp*, const _Tp*>
    : __is_nonvolatile_trivially_copyable<_Tp>
    { };




  template<typename _Tp, bool _TreatAsBytes = __is_byte<_Tp>::__value>
    struct __is_memcmp_ordered
    {
      static const bool __value = _Tp(-1) > _Tp(1);
    };

  template<typename _Tp>
    struct __is_memcmp_ordered<_Tp, false>
    {
      static const bool __value = false;
    };


  template<typename _Tp, typename _Up, bool = sizeof(_Tp) == sizeof(_Up)>
    struct __is_memcmp_ordered_with
    {
      static const bool __value = __is_memcmp_ordered<_Tp>::__value
 && __is_memcmp_ordered<_Up>::__value;
    };

  template<typename _Tp, typename _Up>
    struct __is_memcmp_ordered_with<_Tp, _Up, false>
    {
      static const bool __value = false;
    };
# 532 "/usr/include/c++/10/bits/cpp_type_traits.h" 3
  template<typename _Tp>
    struct __is_move_iterator
    {
      enum { __value = 0 };
      typedef __false_type __type;
    };



  template<typename _Iterator>
   
    inline _Iterator
    __miter_base(_Iterator __it)
    { return __it; }


}
}
# 43 "/usr/include/c++/10/cmath" 2 3
# 1 "/usr/include/c++/10/ext/type_traits.h" 1 3
# 32 "/usr/include/c++/10/ext/type_traits.h" 3
       
# 33 "/usr/include/c++/10/ext/type_traits.h" 3




extern "C++" {

namespace __gnu_cxx __attribute__ ((__visibility__ ("default")))
{



  template<bool, typename>
    struct __enable_if
    { };

  template<typename _Tp>
    struct __enable_if<true, _Tp>
    { typedef _Tp __type; };



  template<bool _Cond, typename _Iftrue, typename _Iffalse>
    struct __conditional_type
    { typedef _Iftrue __type; };

  template<typename _Iftrue, typename _Iffalse>
    struct __conditional_type<false, _Iftrue, _Iffalse>
    { typedef _Iffalse __type; };



  template<typename _Tp>
    struct __add_unsigned
    {
    private:
      typedef __enable_if<std::__is_integer<_Tp>::__value, _Tp> __if_type;

    public:
      typedef typename __if_type::__type __type;
    };

  template<>
    struct __add_unsigned<char>
    { typedef unsigned char __type; };

  template<>
    struct __add_unsigned<signed char>
    { typedef unsigned char __type; };

  template<>
    struct __add_unsigned<short>
    { typedef unsigned short __type; };

  template<>
    struct __add_unsigned<int>
    { typedef unsigned int __type; };

  template<>
    struct __add_unsigned<long>
    { typedef unsigned long __type; };

  template<>
    struct __add_unsigned<long long>
    { typedef unsigned long long __type; };


  template<>
    struct __add_unsigned<bool>;

  template<>
    struct __add_unsigned<wchar_t>;



  template<typename _Tp>
    struct __remove_unsigned
    {
    private:
      typedef __enable_if<std::__is_integer<_Tp>::__value, _Tp> __if_type;

    public:
      typedef typename __if_type::__type __type;
    };

  template<>
    struct __remove_unsigned<char>
    { typedef signed char __type; };

  template<>
    struct __remove_unsigned<unsigned char>
    { typedef signed char __type; };

  template<>
    struct __remove_unsigned<unsigned short>
    { typedef short __type; };

  template<>
    struct __remove_unsigned<unsigned int>
    { typedef int __type; };

  template<>
    struct __remove_unsigned<unsigned long>
    { typedef long __type; };

  template<>
    struct __remove_unsigned<unsigned long long>
    { typedef long long __type; };


  template<>
    struct __remove_unsigned<bool>;

  template<>
    struct __remove_unsigned<wchar_t>;



  template<typename _Type>
    inline bool
    __is_null_pointer(_Type* __ptr)
    { return __ptr == 0; }

  template<typename _Type>
    inline bool
    __is_null_pointer(_Type)
    { return false; }


  inline bool
  __is_null_pointer(std::nullptr_t)
  { return true; }



  template<typename _Tp, bool = std::__is_integer<_Tp>::__value>
    struct __promote
    { typedef double __type; };




  template<typename _Tp>
    struct __promote<_Tp, false>
    { };

  template<>
    struct __promote<long double>
    { typedef long double __type; };

  template<>
    struct __promote<double>
    { typedef double __type; };

  template<>
    struct __promote<float>
    { typedef float __type; };

  template<typename _Tp, typename _Up,
           typename _Tp2 = typename __promote<_Tp>::__type,
           typename _Up2 = typename __promote<_Up>::__type>
    struct __promote_2
    {
      typedef __typeof__(_Tp2() + _Up2()) __type;
    };

  template<typename _Tp, typename _Up, typename _Vp,
           typename _Tp2 = typename __promote<_Tp>::__type,
           typename _Up2 = typename __promote<_Up>::__type,
           typename _Vp2 = typename __promote<_Vp>::__type>
    struct __promote_3
    {
      typedef __typeof__(_Tp2() + _Up2() + _Vp2()) __type;
    };

  template<typename _Tp, typename _Up, typename _Vp, typename _Wp,
           typename _Tp2 = typename __promote<_Tp>::__type,
           typename _Up2 = typename __promote<_Up>::__type,
           typename _Vp2 = typename __promote<_Vp>::__type,
           typename _Wp2 = typename __promote<_Wp>::__type>
    struct __promote_4
    {
      typedef __typeof__(_Tp2() + _Up2() + _Vp2() + _Wp2()) __type;
    };


}
}
# 44 "/usr/include/c++/10/cmath" 2 3

# 1 "/usr/include/math.h" 1 3 4
# 27 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 28 "/usr/include/math.h" 2 3 4






extern "C" {





# 1 "/usr/include/bits/math-vector.h" 1 3 4
# 25 "/usr/include/bits/math-vector.h" 3 4
# 1 "/usr/include/bits/libm-simd-decl-stubs.h" 1 3 4
# 26 "/usr/include/bits/math-vector.h" 2 3 4
# 41 "/usr/include/math.h" 2 3 4
# 138 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/flt-eval-method.h" 1 3 4
# 139 "/usr/include/math.h" 2 3 4
# 149 "/usr/include/math.h" 3 4
typedef float float_t;
typedef double double_t;
# 190 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/fp-logb.h" 1 3 4
# 191 "/usr/include/math.h" 2 3 4
# 233 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/fp-fast.h" 1 3 4
# 234 "/usr/include/math.h" 2 3 4



enum
  {
    FP_INT_UPWARD =

      0,
    FP_INT_DOWNWARD =

      1,
    FP_INT_TOWARDZERO =

      2,
    FP_INT_TONEARESTFROMZERO =

      3,
    FP_INT_TONEAREST =

      4,
  };
# 298 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-helper-functions.h" 1 3 4
# 20 "/usr/include/bits/mathcalls-helper-functions.h" 3 4
extern int __fpclassify (double __value) throw ()
     __attribute__ ((__const__));


extern int __signbit (double __value) throw ()
     __attribute__ ((__const__));



extern int __isinf (double __value) throw ()
  __attribute__ ((__const__));


extern int __finite (double __value) throw ()
  __attribute__ ((__const__));


extern int __isnan (double __value) throw ()
  __attribute__ ((__const__));


extern int __iseqsig (double __x, double __y) throw ();


extern int __issignaling (double __value) throw ()
     __attribute__ ((__const__));
# 299 "/usr/include/math.h" 2 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 53 "/usr/include/bits/mathcalls.h" 3 4
extern double acos (double __x) throw (); extern double __acos (double __x) throw ();

extern double asin (double __x) throw (); extern double __asin (double __x) throw ();

extern double atan (double __x) throw (); extern double __atan (double __x) throw ();

extern double atan2 (double __y, double __x) throw (); extern double __atan2 (double __y, double __x) throw ();


 extern double cos (double __x) throw (); extern double __cos (double __x) throw ();

 extern double sin (double __x) throw (); extern double __sin (double __x) throw ();

extern double tan (double __x) throw (); extern double __tan (double __x) throw ();




extern double cosh (double __x) throw (); extern double __cosh (double __x) throw ();

extern double sinh (double __x) throw (); extern double __sinh (double __x) throw ();

extern double tanh (double __x) throw (); extern double __tanh (double __x) throw ();



 extern void sincos (double __x, double *__sinx, double *__cosx) throw (); extern void __sincos (double __x, double *__sinx, double *__cosx) throw ()
                                                        ;




extern double acosh (double __x) throw (); extern double __acosh (double __x) throw ();

extern double asinh (double __x) throw (); extern double __asinh (double __x) throw ();

extern double atanh (double __x) throw (); extern double __atanh (double __x) throw ();





 extern double exp (double __x) throw (); extern double __exp (double __x) throw ();


extern double frexp (double __x, int *__exponent) throw (); extern double __frexp (double __x, int *__exponent) throw ();


extern double ldexp (double __x, int __exponent) throw (); extern double __ldexp (double __x, int __exponent) throw ();


 extern double log (double __x) throw (); extern double __log (double __x) throw ();


extern double log10 (double __x) throw (); extern double __log10 (double __x) throw ();


extern double modf (double __x, double *__iptr) throw (); extern double __modf (double __x, double *__iptr) throw () __attribute__ ((__nonnull__ (2)));



extern double exp10 (double __x) throw (); extern double __exp10 (double __x) throw ();




extern double expm1 (double __x) throw (); extern double __expm1 (double __x) throw ();


extern double log1p (double __x) throw (); extern double __log1p (double __x) throw ();


extern double logb (double __x) throw (); extern double __logb (double __x) throw ();




extern double exp2 (double __x) throw (); extern double __exp2 (double __x) throw ();


extern double log2 (double __x) throw (); extern double __log2 (double __x) throw ();






 extern double pow (double __x, double __y) throw (); extern double __pow (double __x, double __y) throw ();


extern double sqrt (double __x) throw (); extern double __sqrt (double __x) throw ();



extern double hypot (double __x, double __y) throw (); extern double __hypot (double __x, double __y) throw ();




extern double cbrt (double __x) throw (); extern double __cbrt (double __x) throw ();






extern double ceil (double __x) throw () __attribute__ ((__const__)); extern double __ceil (double __x) throw () __attribute__ ((__const__));


extern double fabs (double __x) throw () __attribute__ ((__const__)); extern double __fabs (double __x) throw () __attribute__ ((__const__));


extern double floor (double __x) throw () __attribute__ ((__const__)); extern double __floor (double __x) throw () __attribute__ ((__const__));


extern double fmod (double __x, double __y) throw (); extern double __fmod (double __x, double __y) throw ();
# 183 "/usr/include/bits/mathcalls.h" 3 4
extern int finite (double __value) throw ()
  __attribute__ ((__const__));


extern double drem (double __x, double __y) throw (); extern double __drem (double __x, double __y) throw ();



extern double significand (double __x) throw (); extern double __significand (double __x) throw ();






extern double copysign (double __x, double __y) throw () __attribute__ ((__const__)); extern double __copysign (double __x, double __y) throw () __attribute__ ((__const__));




extern double nan (const char *__tagb) throw (); extern double __nan (const char *__tagb) throw ();
# 220 "/usr/include/bits/mathcalls.h" 3 4
extern double j0 (double) throw (); extern double __j0 (double) throw ();
extern double j1 (double) throw (); extern double __j1 (double) throw ();
extern double jn (int, double) throw (); extern double __jn (int, double) throw ();
extern double y0 (double) throw (); extern double __y0 (double) throw ();
extern double y1 (double) throw (); extern double __y1 (double) throw ();
extern double yn (int, double) throw (); extern double __yn (int, double) throw ();





extern double erf (double) throw (); extern double __erf (double) throw ();
extern double erfc (double) throw (); extern double __erfc (double) throw ();
extern double lgamma (double) throw (); extern double __lgamma (double) throw ();




extern double tgamma (double) throw (); extern double __tgamma (double) throw ();





extern double gamma (double) throw (); extern double __gamma (double) throw ();







extern double lgamma_r (double, int *__signgamp) throw (); extern double __lgamma_r (double, int *__signgamp) throw ();






extern double rint (double __x) throw (); extern double __rint (double __x) throw ();


extern double nextafter (double __x, double __y) throw (); extern double __nextafter (double __x, double __y) throw ();

extern double nexttoward (double __x, long double __y) throw (); extern double __nexttoward (double __x, long double __y) throw ();




extern double nextdown (double __x) throw (); extern double __nextdown (double __x) throw ();

extern double nextup (double __x) throw (); extern double __nextup (double __x) throw ();



extern double remainder (double __x, double __y) throw (); extern double __remainder (double __x, double __y) throw ();



extern double scalbn (double __x, int __n) throw (); extern double __scalbn (double __x, int __n) throw ();



extern int ilogb (double __x) throw (); extern int __ilogb (double __x) throw ();




extern long int llogb (double __x) throw (); extern long int __llogb (double __x) throw ();




extern double scalbln (double __x, long int __n) throw (); extern double __scalbln (double __x, long int __n) throw ();



extern double nearbyint (double __x) throw (); extern double __nearbyint (double __x) throw ();



extern double round (double __x) throw () __attribute__ ((__const__)); extern double __round (double __x) throw () __attribute__ ((__const__));



extern double trunc (double __x) throw () __attribute__ ((__const__)); extern double __trunc (double __x) throw () __attribute__ ((__const__));




extern double remquo (double __x, double __y, int *__quo) throw (); extern double __remquo (double __x, double __y, int *__quo) throw ();






extern long int lrint (double __x) throw (); extern long int __lrint (double __x) throw ();
__extension__
extern long long int llrint (double __x) throw (); extern long long int __llrint (double __x) throw ();



extern long int lround (double __x) throw (); extern long int __lround (double __x) throw ();
__extension__
extern long long int llround (double __x) throw (); extern long long int __llround (double __x) throw ();



extern double fdim (double __x, double __y) throw (); extern double __fdim (double __x, double __y) throw ();


extern double fmax (double __x, double __y) throw () __attribute__ ((__const__)); extern double __fmax (double __x, double __y) throw () __attribute__ ((__const__));


extern double fmin (double __x, double __y) throw () __attribute__ ((__const__)); extern double __fmin (double __x, double __y) throw () __attribute__ ((__const__));


extern double fma (double __x, double __y, double __z) throw (); extern double __fma (double __x, double __y, double __z) throw ();




extern double roundeven (double __x) throw () __attribute__ ((__const__)); extern double __roundeven (double __x) throw () __attribute__ ((__const__));



extern __intmax_t fromfp (double __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfp (double __x, int __round, unsigned int __width) throw ()
                            ;



extern __uintmax_t ufromfp (double __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfp (double __x, int __round, unsigned int __width) throw ()
                              ;




extern __intmax_t fromfpx (double __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpx (double __x, int __round, unsigned int __width) throw ()
                             ;




extern __uintmax_t ufromfpx (double __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpx (double __x, int __round, unsigned int __width) throw ()
                               ;


extern double fmaxmag (double __x, double __y) throw () __attribute__ ((__const__)); extern double __fmaxmag (double __x, double __y) throw () __attribute__ ((__const__));


extern double fminmag (double __x, double __y) throw () __attribute__ ((__const__)); extern double __fminmag (double __x, double __y) throw () __attribute__ ((__const__));


extern int canonicalize (double *__cx, const double *__x) throw ();




extern int totalorder (const double *__x, const double *__y) throw ()

     __attribute__ ((__pure__));


extern int totalordermag (const double *__x, const double *__y) throw ()

     __attribute__ ((__pure__));


extern double getpayload (const double *__x) throw (); extern double __getpayload (const double *__x) throw ();


extern int setpayload (double *__x, double __payload) throw ();


extern int setpayloadsig (double *__x, double __payload) throw ();







extern double scalb (double __x, double __n) throw (); extern double __scalb (double __x, double __n) throw ();
# 300 "/usr/include/math.h" 2 3 4
# 315 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-helper-functions.h" 1 3 4
# 20 "/usr/include/bits/mathcalls-helper-functions.h" 3 4
extern int __fpclassifyf (float __value) throw ()
     __attribute__ ((__const__));


extern int __signbitf (float __value) throw ()
     __attribute__ ((__const__));



extern int __isinff (float __value) throw ()
  __attribute__ ((__const__));


extern int __finitef (float __value) throw ()
  __attribute__ ((__const__));


extern int __isnanf (float __value) throw ()
  __attribute__ ((__const__));


extern int __iseqsigf (float __x, float __y) throw ();


extern int __issignalingf (float __value) throw ()
     __attribute__ ((__const__));
# 316 "/usr/include/math.h" 2 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 53 "/usr/include/bits/mathcalls.h" 3 4
extern float acosf (float __x) throw (); extern float __acosf (float __x) throw ();

extern float asinf (float __x) throw (); extern float __asinf (float __x) throw ();

extern float atanf (float __x) throw (); extern float __atanf (float __x) throw ();

extern float atan2f (float __y, float __x) throw (); extern float __atan2f (float __y, float __x) throw ();


 extern float cosf (float __x) throw (); extern float __cosf (float __x) throw ();

 extern float sinf (float __x) throw (); extern float __sinf (float __x) throw ();

extern float tanf (float __x) throw (); extern float __tanf (float __x) throw ();




extern float coshf (float __x) throw (); extern float __coshf (float __x) throw ();

extern float sinhf (float __x) throw (); extern float __sinhf (float __x) throw ();

extern float tanhf (float __x) throw (); extern float __tanhf (float __x) throw ();



 extern void sincosf (float __x, float *__sinx, float *__cosx) throw (); extern void __sincosf (float __x, float *__sinx, float *__cosx) throw ()
                                                        ;




extern float acoshf (float __x) throw (); extern float __acoshf (float __x) throw ();

extern float asinhf (float __x) throw (); extern float __asinhf (float __x) throw ();

extern float atanhf (float __x) throw (); extern float __atanhf (float __x) throw ();





 extern float expf (float __x) throw (); extern float __expf (float __x) throw ();


extern float frexpf (float __x, int *__exponent) throw (); extern float __frexpf (float __x, int *__exponent) throw ();


extern float ldexpf (float __x, int __exponent) throw (); extern float __ldexpf (float __x, int __exponent) throw ();


 extern float logf (float __x) throw (); extern float __logf (float __x) throw ();


extern float log10f (float __x) throw (); extern float __log10f (float __x) throw ();


extern float modff (float __x, float *__iptr) throw (); extern float __modff (float __x, float *__iptr) throw () __attribute__ ((__nonnull__ (2)));



extern float exp10f (float __x) throw (); extern float __exp10f (float __x) throw ();




extern float expm1f (float __x) throw (); extern float __expm1f (float __x) throw ();


extern float log1pf (float __x) throw (); extern float __log1pf (float __x) throw ();


extern float logbf (float __x) throw (); extern float __logbf (float __x) throw ();




extern float exp2f (float __x) throw (); extern float __exp2f (float __x) throw ();


extern float log2f (float __x) throw (); extern float __log2f (float __x) throw ();






 extern float powf (float __x, float __y) throw (); extern float __powf (float __x, float __y) throw ();


extern float sqrtf (float __x) throw (); extern float __sqrtf (float __x) throw ();



extern float hypotf (float __x, float __y) throw (); extern float __hypotf (float __x, float __y) throw ();




extern float cbrtf (float __x) throw (); extern float __cbrtf (float __x) throw ();






extern float ceilf (float __x) throw () __attribute__ ((__const__)); extern float __ceilf (float __x) throw () __attribute__ ((__const__));


extern float fabsf (float __x) throw () __attribute__ ((__const__)); extern float __fabsf (float __x) throw () __attribute__ ((__const__));


extern float floorf (float __x) throw () __attribute__ ((__const__)); extern float __floorf (float __x) throw () __attribute__ ((__const__));


extern float fmodf (float __x, float __y) throw (); extern float __fmodf (float __x, float __y) throw ();
# 177 "/usr/include/bits/mathcalls.h" 3 4
extern int isinff (float __value) throw ()
  __attribute__ ((__const__));




extern int finitef (float __value) throw ()
  __attribute__ ((__const__));


extern float dremf (float __x, float __y) throw (); extern float __dremf (float __x, float __y) throw ();



extern float significandf (float __x) throw (); extern float __significandf (float __x) throw ();






extern float copysignf (float __x, float __y) throw () __attribute__ ((__const__)); extern float __copysignf (float __x, float __y) throw () __attribute__ ((__const__));




extern float nanf (const char *__tagb) throw (); extern float __nanf (const char *__tagb) throw ();
# 213 "/usr/include/bits/mathcalls.h" 3 4
extern int isnanf (float __value) throw ()
  __attribute__ ((__const__));





extern float j0f (float) throw (); extern float __j0f (float) throw ();
extern float j1f (float) throw (); extern float __j1f (float) throw ();
extern float jnf (int, float) throw (); extern float __jnf (int, float) throw ();
extern float y0f (float) throw (); extern float __y0f (float) throw ();
extern float y1f (float) throw (); extern float __y1f (float) throw ();
extern float ynf (int, float) throw (); extern float __ynf (int, float) throw ();





extern float erff (float) throw (); extern float __erff (float) throw ();
extern float erfcf (float) throw (); extern float __erfcf (float) throw ();
extern float lgammaf (float) throw (); extern float __lgammaf (float) throw ();




extern float tgammaf (float) throw (); extern float __tgammaf (float) throw ();





extern float gammaf (float) throw (); extern float __gammaf (float) throw ();







extern float lgammaf_r (float, int *__signgamp) throw (); extern float __lgammaf_r (float, int *__signgamp) throw ();






extern float rintf (float __x) throw (); extern float __rintf (float __x) throw ();


extern float nextafterf (float __x, float __y) throw (); extern float __nextafterf (float __x, float __y) throw ();

extern float nexttowardf (float __x, long double __y) throw (); extern float __nexttowardf (float __x, long double __y) throw ();




extern float nextdownf (float __x) throw (); extern float __nextdownf (float __x) throw ();

extern float nextupf (float __x) throw (); extern float __nextupf (float __x) throw ();



extern float remainderf (float __x, float __y) throw (); extern float __remainderf (float __x, float __y) throw ();



extern float scalbnf (float __x, int __n) throw (); extern float __scalbnf (float __x, int __n) throw ();



extern int ilogbf (float __x) throw (); extern int __ilogbf (float __x) throw ();




extern long int llogbf (float __x) throw (); extern long int __llogbf (float __x) throw ();




extern float scalblnf (float __x, long int __n) throw (); extern float __scalblnf (float __x, long int __n) throw ();



extern float nearbyintf (float __x) throw (); extern float __nearbyintf (float __x) throw ();



extern float roundf (float __x) throw () __attribute__ ((__const__)); extern float __roundf (float __x) throw () __attribute__ ((__const__));



extern float truncf (float __x) throw () __attribute__ ((__const__)); extern float __truncf (float __x) throw () __attribute__ ((__const__));




extern float remquof (float __x, float __y, int *__quo) throw (); extern float __remquof (float __x, float __y, int *__quo) throw ();






extern long int lrintf (float __x) throw (); extern long int __lrintf (float __x) throw ();
__extension__
extern long long int llrintf (float __x) throw (); extern long long int __llrintf (float __x) throw ();



extern long int lroundf (float __x) throw (); extern long int __lroundf (float __x) throw ();
__extension__
extern long long int llroundf (float __x) throw (); extern long long int __llroundf (float __x) throw ();



extern float fdimf (float __x, float __y) throw (); extern float __fdimf (float __x, float __y) throw ();


extern float fmaxf (float __x, float __y) throw () __attribute__ ((__const__)); extern float __fmaxf (float __x, float __y) throw () __attribute__ ((__const__));


extern float fminf (float __x, float __y) throw () __attribute__ ((__const__)); extern float __fminf (float __x, float __y) throw () __attribute__ ((__const__));


extern float fmaf (float __x, float __y, float __z) throw (); extern float __fmaf (float __x, float __y, float __z) throw ();




extern float roundevenf (float __x) throw () __attribute__ ((__const__)); extern float __roundevenf (float __x) throw () __attribute__ ((__const__));



extern __intmax_t fromfpf (float __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpf (float __x, int __round, unsigned int __width) throw ()
                            ;



extern __uintmax_t ufromfpf (float __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpf (float __x, int __round, unsigned int __width) throw ()
                              ;




extern __intmax_t fromfpxf (float __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpxf (float __x, int __round, unsigned int __width) throw ()
                             ;




extern __uintmax_t ufromfpxf (float __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpxf (float __x, int __round, unsigned int __width) throw ()
                               ;


extern float fmaxmagf (float __x, float __y) throw () __attribute__ ((__const__)); extern float __fmaxmagf (float __x, float __y) throw () __attribute__ ((__const__));


extern float fminmagf (float __x, float __y) throw () __attribute__ ((__const__)); extern float __fminmagf (float __x, float __y) throw () __attribute__ ((__const__));


extern int canonicalizef (float *__cx, const float *__x) throw ();




extern int totalorderf (const float *__x, const float *__y) throw ()

     __attribute__ ((__pure__));


extern int totalordermagf (const float *__x, const float *__y) throw ()

     __attribute__ ((__pure__));


extern float getpayloadf (const float *__x) throw (); extern float __getpayloadf (const float *__x) throw ();


extern int setpayloadf (float *__x, float __payload) throw ();


extern int setpayloadsigf (float *__x, float __payload) throw ();







extern float scalbf (float __x, float __n) throw (); extern float __scalbf (float __x, float __n) throw ();
# 317 "/usr/include/math.h" 2 3 4
# 384 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-helper-functions.h" 1 3 4
# 20 "/usr/include/bits/mathcalls-helper-functions.h" 3 4
extern int __fpclassifyl (long double __value) throw ()
     __attribute__ ((__const__));


extern int __signbitl (long double __value) throw ()
     __attribute__ ((__const__));



extern int __isinfl (long double __value) throw ()
  __attribute__ ((__const__));


extern int __finitel (long double __value) throw ()
  __attribute__ ((__const__));


extern int __isnanl (long double __value) throw ()
  __attribute__ ((__const__));


extern int __iseqsigl (long double __x, long double __y) throw ();


extern int __issignalingl (long double __value) throw ()
     __attribute__ ((__const__));
# 385 "/usr/include/math.h" 2 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 53 "/usr/include/bits/mathcalls.h" 3 4
extern long double acosl (long double __x) throw (); extern long double __acosl (long double __x) throw ();

extern long double asinl (long double __x) throw (); extern long double __asinl (long double __x) throw ();

extern long double atanl (long double __x) throw (); extern long double __atanl (long double __x) throw ();

extern long double atan2l (long double __y, long double __x) throw (); extern long double __atan2l (long double __y, long double __x) throw ();


 extern long double cosl (long double __x) throw (); extern long double __cosl (long double __x) throw ();

 extern long double sinl (long double __x) throw (); extern long double __sinl (long double __x) throw ();

extern long double tanl (long double __x) throw (); extern long double __tanl (long double __x) throw ();




extern long double coshl (long double __x) throw (); extern long double __coshl (long double __x) throw ();

extern long double sinhl (long double __x) throw (); extern long double __sinhl (long double __x) throw ();

extern long double tanhl (long double __x) throw (); extern long double __tanhl (long double __x) throw ();



 extern void sincosl (long double __x, long double *__sinx, long double *__cosx) throw (); extern void __sincosl (long double __x, long double *__sinx, long double *__cosx) throw ()
                                                        ;




extern long double acoshl (long double __x) throw (); extern long double __acoshl (long double __x) throw ();

extern long double asinhl (long double __x) throw (); extern long double __asinhl (long double __x) throw ();

extern long double atanhl (long double __x) throw (); extern long double __atanhl (long double __x) throw ();





 extern long double expl (long double __x) throw (); extern long double __expl (long double __x) throw ();


extern long double frexpl (long double __x, int *__exponent) throw (); extern long double __frexpl (long double __x, int *__exponent) throw ();


extern long double ldexpl (long double __x, int __exponent) throw (); extern long double __ldexpl (long double __x, int __exponent) throw ();


 extern long double logl (long double __x) throw (); extern long double __logl (long double __x) throw ();


extern long double log10l (long double __x) throw (); extern long double __log10l (long double __x) throw ();


extern long double modfl (long double __x, long double *__iptr) throw (); extern long double __modfl (long double __x, long double *__iptr) throw () __attribute__ ((__nonnull__ (2)));



extern long double exp10l (long double __x) throw (); extern long double __exp10l (long double __x) throw ();




extern long double expm1l (long double __x) throw (); extern long double __expm1l (long double __x) throw ();


extern long double log1pl (long double __x) throw (); extern long double __log1pl (long double __x) throw ();


extern long double logbl (long double __x) throw (); extern long double __logbl (long double __x) throw ();




extern long double exp2l (long double __x) throw (); extern long double __exp2l (long double __x) throw ();


extern long double log2l (long double __x) throw (); extern long double __log2l (long double __x) throw ();






 extern long double powl (long double __x, long double __y) throw (); extern long double __powl (long double __x, long double __y) throw ();


extern long double sqrtl (long double __x) throw (); extern long double __sqrtl (long double __x) throw ();



extern long double hypotl (long double __x, long double __y) throw (); extern long double __hypotl (long double __x, long double __y) throw ();




extern long double cbrtl (long double __x) throw (); extern long double __cbrtl (long double __x) throw ();






extern long double ceill (long double __x) throw () __attribute__ ((__const__)); extern long double __ceill (long double __x) throw () __attribute__ ((__const__));


extern long double fabsl (long double __x) throw () __attribute__ ((__const__)); extern long double __fabsl (long double __x) throw () __attribute__ ((__const__));


extern long double floorl (long double __x) throw () __attribute__ ((__const__)); extern long double __floorl (long double __x) throw () __attribute__ ((__const__));


extern long double fmodl (long double __x, long double __y) throw (); extern long double __fmodl (long double __x, long double __y) throw ();
# 177 "/usr/include/bits/mathcalls.h" 3 4
extern int isinfl (long double __value) throw ()
  __attribute__ ((__const__));




extern int finitel (long double __value) throw ()
  __attribute__ ((__const__));


extern long double dreml (long double __x, long double __y) throw (); extern long double __dreml (long double __x, long double __y) throw ();



extern long double significandl (long double __x) throw (); extern long double __significandl (long double __x) throw ();






extern long double copysignl (long double __x, long double __y) throw () __attribute__ ((__const__)); extern long double __copysignl (long double __x, long double __y) throw () __attribute__ ((__const__));




extern long double nanl (const char *__tagb) throw (); extern long double __nanl (const char *__tagb) throw ();
# 213 "/usr/include/bits/mathcalls.h" 3 4
extern int isnanl (long double __value) throw ()
  __attribute__ ((__const__));





extern long double j0l (long double) throw (); extern long double __j0l (long double) throw ();
extern long double j1l (long double) throw (); extern long double __j1l (long double) throw ();
extern long double jnl (int, long double) throw (); extern long double __jnl (int, long double) throw ();
extern long double y0l (long double) throw (); extern long double __y0l (long double) throw ();
extern long double y1l (long double) throw (); extern long double __y1l (long double) throw ();
extern long double ynl (int, long double) throw (); extern long double __ynl (int, long double) throw ();





extern long double erfl (long double) throw (); extern long double __erfl (long double) throw ();
extern long double erfcl (long double) throw (); extern long double __erfcl (long double) throw ();
extern long double lgammal (long double) throw (); extern long double __lgammal (long double) throw ();




extern long double tgammal (long double) throw (); extern long double __tgammal (long double) throw ();





extern long double gammal (long double) throw (); extern long double __gammal (long double) throw ();







extern long double lgammal_r (long double, int *__signgamp) throw (); extern long double __lgammal_r (long double, int *__signgamp) throw ();






extern long double rintl (long double __x) throw (); extern long double __rintl (long double __x) throw ();


extern long double nextafterl (long double __x, long double __y) throw (); extern long double __nextafterl (long double __x, long double __y) throw ();

extern long double nexttowardl (long double __x, long double __y) throw (); extern long double __nexttowardl (long double __x, long double __y) throw ();




extern long double nextdownl (long double __x) throw (); extern long double __nextdownl (long double __x) throw ();

extern long double nextupl (long double __x) throw (); extern long double __nextupl (long double __x) throw ();



extern long double remainderl (long double __x, long double __y) throw (); extern long double __remainderl (long double __x, long double __y) throw ();



extern long double scalbnl (long double __x, int __n) throw (); extern long double __scalbnl (long double __x, int __n) throw ();



extern int ilogbl (long double __x) throw (); extern int __ilogbl (long double __x) throw ();




extern long int llogbl (long double __x) throw (); extern long int __llogbl (long double __x) throw ();




extern long double scalblnl (long double __x, long int __n) throw (); extern long double __scalblnl (long double __x, long int __n) throw ();



extern long double nearbyintl (long double __x) throw (); extern long double __nearbyintl (long double __x) throw ();



extern long double roundl (long double __x) throw () __attribute__ ((__const__)); extern long double __roundl (long double __x) throw () __attribute__ ((__const__));



extern long double truncl (long double __x) throw () __attribute__ ((__const__)); extern long double __truncl (long double __x) throw () __attribute__ ((__const__));




extern long double remquol (long double __x, long double __y, int *__quo) throw (); extern long double __remquol (long double __x, long double __y, int *__quo) throw ();






extern long int lrintl (long double __x) throw (); extern long int __lrintl (long double __x) throw ();
__extension__
extern long long int llrintl (long double __x) throw (); extern long long int __llrintl (long double __x) throw ();



extern long int lroundl (long double __x) throw (); extern long int __lroundl (long double __x) throw ();
__extension__
extern long long int llroundl (long double __x) throw (); extern long long int __llroundl (long double __x) throw ();



extern long double fdiml (long double __x, long double __y) throw (); extern long double __fdiml (long double __x, long double __y) throw ();


extern long double fmaxl (long double __x, long double __y) throw () __attribute__ ((__const__)); extern long double __fmaxl (long double __x, long double __y) throw () __attribute__ ((__const__));


extern long double fminl (long double __x, long double __y) throw () __attribute__ ((__const__)); extern long double __fminl (long double __x, long double __y) throw () __attribute__ ((__const__));


extern long double fmal (long double __x, long double __y, long double __z) throw (); extern long double __fmal (long double __x, long double __y, long double __z) throw ();




extern long double roundevenl (long double __x) throw () __attribute__ ((__const__)); extern long double __roundevenl (long double __x) throw () __attribute__ ((__const__));



extern __intmax_t fromfpl (long double __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpl (long double __x, int __round, unsigned int __width) throw ()
                            ;



extern __uintmax_t ufromfpl (long double __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpl (long double __x, int __round, unsigned int __width) throw ()
                              ;




extern __intmax_t fromfpxl (long double __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpxl (long double __x, int __round, unsigned int __width) throw ()
                             ;




extern __uintmax_t ufromfpxl (long double __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpxl (long double __x, int __round, unsigned int __width) throw ()
                               ;


extern long double fmaxmagl (long double __x, long double __y) throw () __attribute__ ((__const__)); extern long double __fmaxmagl (long double __x, long double __y) throw () __attribute__ ((__const__));


extern long double fminmagl (long double __x, long double __y) throw () __attribute__ ((__const__)); extern long double __fminmagl (long double __x, long double __y) throw () __attribute__ ((__const__));


extern int canonicalizel (long double *__cx, const long double *__x) throw ();




extern int totalorderl (const long double *__x, const long double *__y) throw ()

     __attribute__ ((__pure__));


extern int totalordermagl (const long double *__x, const long double *__y) throw ()

     __attribute__ ((__pure__));


extern long double getpayloadl (const long double *__x) throw (); extern long double __getpayloadl (const long double *__x) throw ();


extern int setpayloadl (long double *__x, long double __payload) throw ();


extern int setpayloadsigl (long double *__x, long double __payload) throw ();







extern long double scalbl (long double __x, long double __n) throw (); extern long double __scalbl (long double __x, long double __n) throw ();
# 386 "/usr/include/math.h" 2 3 4
# 436 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 53 "/usr/include/bits/mathcalls.h" 3 4
extern _Float32 acosf32 (_Float32 __x) throw (); extern _Float32 __acosf32 (_Float32 __x) throw ();

extern _Float32 asinf32 (_Float32 __x) throw (); extern _Float32 __asinf32 (_Float32 __x) throw ();

extern _Float32 atanf32 (_Float32 __x) throw (); extern _Float32 __atanf32 (_Float32 __x) throw ();

extern _Float32 atan2f32 (_Float32 __y, _Float32 __x) throw (); extern _Float32 __atan2f32 (_Float32 __y, _Float32 __x) throw ();


 extern _Float32 cosf32 (_Float32 __x) throw (); extern _Float32 __cosf32 (_Float32 __x) throw ();

 extern _Float32 sinf32 (_Float32 __x) throw (); extern _Float32 __sinf32 (_Float32 __x) throw ();

extern _Float32 tanf32 (_Float32 __x) throw (); extern _Float32 __tanf32 (_Float32 __x) throw ();




extern _Float32 coshf32 (_Float32 __x) throw (); extern _Float32 __coshf32 (_Float32 __x) throw ();

extern _Float32 sinhf32 (_Float32 __x) throw (); extern _Float32 __sinhf32 (_Float32 __x) throw ();

extern _Float32 tanhf32 (_Float32 __x) throw (); extern _Float32 __tanhf32 (_Float32 __x) throw ();



 extern void sincosf32 (_Float32 __x, _Float32 *__sinx, _Float32 *__cosx) throw (); extern void __sincosf32 (_Float32 __x, _Float32 *__sinx, _Float32 *__cosx) throw ()
                                                        ;




extern _Float32 acoshf32 (_Float32 __x) throw (); extern _Float32 __acoshf32 (_Float32 __x) throw ();

extern _Float32 asinhf32 (_Float32 __x) throw (); extern _Float32 __asinhf32 (_Float32 __x) throw ();

extern _Float32 atanhf32 (_Float32 __x) throw (); extern _Float32 __atanhf32 (_Float32 __x) throw ();





 extern _Float32 expf32 (_Float32 __x) throw (); extern _Float32 __expf32 (_Float32 __x) throw ();


extern _Float32 frexpf32 (_Float32 __x, int *__exponent) throw (); extern _Float32 __frexpf32 (_Float32 __x, int *__exponent) throw ();


extern _Float32 ldexpf32 (_Float32 __x, int __exponent) throw (); extern _Float32 __ldexpf32 (_Float32 __x, int __exponent) throw ();


 extern _Float32 logf32 (_Float32 __x) throw (); extern _Float32 __logf32 (_Float32 __x) throw ();


extern _Float32 log10f32 (_Float32 __x) throw (); extern _Float32 __log10f32 (_Float32 __x) throw ();


extern _Float32 modff32 (_Float32 __x, _Float32 *__iptr) throw (); extern _Float32 __modff32 (_Float32 __x, _Float32 *__iptr) throw () __attribute__ ((__nonnull__ (2)));



extern _Float32 exp10f32 (_Float32 __x) throw (); extern _Float32 __exp10f32 (_Float32 __x) throw ();




extern _Float32 expm1f32 (_Float32 __x) throw (); extern _Float32 __expm1f32 (_Float32 __x) throw ();


extern _Float32 log1pf32 (_Float32 __x) throw (); extern _Float32 __log1pf32 (_Float32 __x) throw ();


extern _Float32 logbf32 (_Float32 __x) throw (); extern _Float32 __logbf32 (_Float32 __x) throw ();




extern _Float32 exp2f32 (_Float32 __x) throw (); extern _Float32 __exp2f32 (_Float32 __x) throw ();


extern _Float32 log2f32 (_Float32 __x) throw (); extern _Float32 __log2f32 (_Float32 __x) throw ();






 extern _Float32 powf32 (_Float32 __x, _Float32 __y) throw (); extern _Float32 __powf32 (_Float32 __x, _Float32 __y) throw ();


extern _Float32 sqrtf32 (_Float32 __x) throw (); extern _Float32 __sqrtf32 (_Float32 __x) throw ();



extern _Float32 hypotf32 (_Float32 __x, _Float32 __y) throw (); extern _Float32 __hypotf32 (_Float32 __x, _Float32 __y) throw ();




extern _Float32 cbrtf32 (_Float32 __x) throw (); extern _Float32 __cbrtf32 (_Float32 __x) throw ();






extern _Float32 ceilf32 (_Float32 __x) throw () __attribute__ ((__const__)); extern _Float32 __ceilf32 (_Float32 __x) throw () __attribute__ ((__const__));


extern _Float32 fabsf32 (_Float32 __x) throw () __attribute__ ((__const__)); extern _Float32 __fabsf32 (_Float32 __x) throw () __attribute__ ((__const__));


extern _Float32 floorf32 (_Float32 __x) throw () __attribute__ ((__const__)); extern _Float32 __floorf32 (_Float32 __x) throw () __attribute__ ((__const__));


extern _Float32 fmodf32 (_Float32 __x, _Float32 __y) throw (); extern _Float32 __fmodf32 (_Float32 __x, _Float32 __y) throw ();
# 198 "/usr/include/bits/mathcalls.h" 3 4
extern _Float32 copysignf32 (_Float32 __x, _Float32 __y) throw () __attribute__ ((__const__)); extern _Float32 __copysignf32 (_Float32 __x, _Float32 __y) throw () __attribute__ ((__const__));




extern _Float32 nanf32 (const char *__tagb) throw (); extern _Float32 __nanf32 (const char *__tagb) throw ();
# 220 "/usr/include/bits/mathcalls.h" 3 4
extern _Float32 j0f32 (_Float32) throw (); extern _Float32 __j0f32 (_Float32) throw ();
extern _Float32 j1f32 (_Float32) throw (); extern _Float32 __j1f32 (_Float32) throw ();
extern _Float32 jnf32 (int, _Float32) throw (); extern _Float32 __jnf32 (int, _Float32) throw ();
extern _Float32 y0f32 (_Float32) throw (); extern _Float32 __y0f32 (_Float32) throw ();
extern _Float32 y1f32 (_Float32) throw (); extern _Float32 __y1f32 (_Float32) throw ();
extern _Float32 ynf32 (int, _Float32) throw (); extern _Float32 __ynf32 (int, _Float32) throw ();





extern _Float32 erff32 (_Float32) throw (); extern _Float32 __erff32 (_Float32) throw ();
extern _Float32 erfcf32 (_Float32) throw (); extern _Float32 __erfcf32 (_Float32) throw ();
extern _Float32 lgammaf32 (_Float32) throw (); extern _Float32 __lgammaf32 (_Float32) throw ();




extern _Float32 tgammaf32 (_Float32) throw (); extern _Float32 __tgammaf32 (_Float32) throw ();
# 252 "/usr/include/bits/mathcalls.h" 3 4
extern _Float32 lgammaf32_r (_Float32, int *__signgamp) throw (); extern _Float32 __lgammaf32_r (_Float32, int *__signgamp) throw ();






extern _Float32 rintf32 (_Float32 __x) throw (); extern _Float32 __rintf32 (_Float32 __x) throw ();


extern _Float32 nextafterf32 (_Float32 __x, _Float32 __y) throw (); extern _Float32 __nextafterf32 (_Float32 __x, _Float32 __y) throw ();






extern _Float32 nextdownf32 (_Float32 __x) throw (); extern _Float32 __nextdownf32 (_Float32 __x) throw ();

extern _Float32 nextupf32 (_Float32 __x) throw (); extern _Float32 __nextupf32 (_Float32 __x) throw ();



extern _Float32 remainderf32 (_Float32 __x, _Float32 __y) throw (); extern _Float32 __remainderf32 (_Float32 __x, _Float32 __y) throw ();



extern _Float32 scalbnf32 (_Float32 __x, int __n) throw (); extern _Float32 __scalbnf32 (_Float32 __x, int __n) throw ();



extern int ilogbf32 (_Float32 __x) throw (); extern int __ilogbf32 (_Float32 __x) throw ();




extern long int llogbf32 (_Float32 __x) throw (); extern long int __llogbf32 (_Float32 __x) throw ();




extern _Float32 scalblnf32 (_Float32 __x, long int __n) throw (); extern _Float32 __scalblnf32 (_Float32 __x, long int __n) throw ();



extern _Float32 nearbyintf32 (_Float32 __x) throw (); extern _Float32 __nearbyintf32 (_Float32 __x) throw ();



extern _Float32 roundf32 (_Float32 __x) throw () __attribute__ ((__const__)); extern _Float32 __roundf32 (_Float32 __x) throw () __attribute__ ((__const__));



extern _Float32 truncf32 (_Float32 __x) throw () __attribute__ ((__const__)); extern _Float32 __truncf32 (_Float32 __x) throw () __attribute__ ((__const__));




extern _Float32 remquof32 (_Float32 __x, _Float32 __y, int *__quo) throw (); extern _Float32 __remquof32 (_Float32 __x, _Float32 __y, int *__quo) throw ();






extern long int lrintf32 (_Float32 __x) throw (); extern long int __lrintf32 (_Float32 __x) throw ();
__extension__
extern long long int llrintf32 (_Float32 __x) throw (); extern long long int __llrintf32 (_Float32 __x) throw ();



extern long int lroundf32 (_Float32 __x) throw (); extern long int __lroundf32 (_Float32 __x) throw ();
__extension__
extern long long int llroundf32 (_Float32 __x) throw (); extern long long int __llroundf32 (_Float32 __x) throw ();



extern _Float32 fdimf32 (_Float32 __x, _Float32 __y) throw (); extern _Float32 __fdimf32 (_Float32 __x, _Float32 __y) throw ();


extern _Float32 fmaxf32 (_Float32 __x, _Float32 __y) throw () __attribute__ ((__const__)); extern _Float32 __fmaxf32 (_Float32 __x, _Float32 __y) throw () __attribute__ ((__const__));


extern _Float32 fminf32 (_Float32 __x, _Float32 __y) throw () __attribute__ ((__const__)); extern _Float32 __fminf32 (_Float32 __x, _Float32 __y) throw () __attribute__ ((__const__));


extern _Float32 fmaf32 (_Float32 __x, _Float32 __y, _Float32 __z) throw (); extern _Float32 __fmaf32 (_Float32 __x, _Float32 __y, _Float32 __z) throw ();




extern _Float32 roundevenf32 (_Float32 __x) throw () __attribute__ ((__const__)); extern _Float32 __roundevenf32 (_Float32 __x) throw () __attribute__ ((__const__));



extern __intmax_t fromfpf32 (_Float32 __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpf32 (_Float32 __x, int __round, unsigned int __width) throw ()
                            ;



extern __uintmax_t ufromfpf32 (_Float32 __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpf32 (_Float32 __x, int __round, unsigned int __width) throw ()
                              ;




extern __intmax_t fromfpxf32 (_Float32 __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpxf32 (_Float32 __x, int __round, unsigned int __width) throw ()
                             ;




extern __uintmax_t ufromfpxf32 (_Float32 __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpxf32 (_Float32 __x, int __round, unsigned int __width) throw ()
                               ;


extern _Float32 fmaxmagf32 (_Float32 __x, _Float32 __y) throw () __attribute__ ((__const__)); extern _Float32 __fmaxmagf32 (_Float32 __x, _Float32 __y) throw () __attribute__ ((__const__));


extern _Float32 fminmagf32 (_Float32 __x, _Float32 __y) throw () __attribute__ ((__const__)); extern _Float32 __fminmagf32 (_Float32 __x, _Float32 __y) throw () __attribute__ ((__const__));


extern int canonicalizef32 (_Float32 *__cx, const _Float32 *__x) throw ();




extern int totalorderf32 (const _Float32 *__x, const _Float32 *__y) throw ()

     __attribute__ ((__pure__));


extern int totalordermagf32 (const _Float32 *__x, const _Float32 *__y) throw ()

     __attribute__ ((__pure__));


extern _Float32 getpayloadf32 (const _Float32 *__x) throw (); extern _Float32 __getpayloadf32 (const _Float32 *__x) throw ();


extern int setpayloadf32 (_Float32 *__x, _Float32 __payload) throw ();


extern int setpayloadsigf32 (_Float32 *__x, _Float32 __payload) throw ();
# 437 "/usr/include/math.h" 2 3 4
# 453 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 53 "/usr/include/bits/mathcalls.h" 3 4
extern _Float64 acosf64 (_Float64 __x) throw (); extern _Float64 __acosf64 (_Float64 __x) throw ();

extern _Float64 asinf64 (_Float64 __x) throw (); extern _Float64 __asinf64 (_Float64 __x) throw ();

extern _Float64 atanf64 (_Float64 __x) throw (); extern _Float64 __atanf64 (_Float64 __x) throw ();

extern _Float64 atan2f64 (_Float64 __y, _Float64 __x) throw (); extern _Float64 __atan2f64 (_Float64 __y, _Float64 __x) throw ();


 extern _Float64 cosf64 (_Float64 __x) throw (); extern _Float64 __cosf64 (_Float64 __x) throw ();

 extern _Float64 sinf64 (_Float64 __x) throw (); extern _Float64 __sinf64 (_Float64 __x) throw ();

extern _Float64 tanf64 (_Float64 __x) throw (); extern _Float64 __tanf64 (_Float64 __x) throw ();




extern _Float64 coshf64 (_Float64 __x) throw (); extern _Float64 __coshf64 (_Float64 __x) throw ();

extern _Float64 sinhf64 (_Float64 __x) throw (); extern _Float64 __sinhf64 (_Float64 __x) throw ();

extern _Float64 tanhf64 (_Float64 __x) throw (); extern _Float64 __tanhf64 (_Float64 __x) throw ();



 extern void sincosf64 (_Float64 __x, _Float64 *__sinx, _Float64 *__cosx) throw (); extern void __sincosf64 (_Float64 __x, _Float64 *__sinx, _Float64 *__cosx) throw ()
                                                        ;




extern _Float64 acoshf64 (_Float64 __x) throw (); extern _Float64 __acoshf64 (_Float64 __x) throw ();

extern _Float64 asinhf64 (_Float64 __x) throw (); extern _Float64 __asinhf64 (_Float64 __x) throw ();

extern _Float64 atanhf64 (_Float64 __x) throw (); extern _Float64 __atanhf64 (_Float64 __x) throw ();





 extern _Float64 expf64 (_Float64 __x) throw (); extern _Float64 __expf64 (_Float64 __x) throw ();


extern _Float64 frexpf64 (_Float64 __x, int *__exponent) throw (); extern _Float64 __frexpf64 (_Float64 __x, int *__exponent) throw ();


extern _Float64 ldexpf64 (_Float64 __x, int __exponent) throw (); extern _Float64 __ldexpf64 (_Float64 __x, int __exponent) throw ();


 extern _Float64 logf64 (_Float64 __x) throw (); extern _Float64 __logf64 (_Float64 __x) throw ();


extern _Float64 log10f64 (_Float64 __x) throw (); extern _Float64 __log10f64 (_Float64 __x) throw ();


extern _Float64 modff64 (_Float64 __x, _Float64 *__iptr) throw (); extern _Float64 __modff64 (_Float64 __x, _Float64 *__iptr) throw () __attribute__ ((__nonnull__ (2)));



extern _Float64 exp10f64 (_Float64 __x) throw (); extern _Float64 __exp10f64 (_Float64 __x) throw ();




extern _Float64 expm1f64 (_Float64 __x) throw (); extern _Float64 __expm1f64 (_Float64 __x) throw ();


extern _Float64 log1pf64 (_Float64 __x) throw (); extern _Float64 __log1pf64 (_Float64 __x) throw ();


extern _Float64 logbf64 (_Float64 __x) throw (); extern _Float64 __logbf64 (_Float64 __x) throw ();




extern _Float64 exp2f64 (_Float64 __x) throw (); extern _Float64 __exp2f64 (_Float64 __x) throw ();


extern _Float64 log2f64 (_Float64 __x) throw (); extern _Float64 __log2f64 (_Float64 __x) throw ();






 extern _Float64 powf64 (_Float64 __x, _Float64 __y) throw (); extern _Float64 __powf64 (_Float64 __x, _Float64 __y) throw ();


extern _Float64 sqrtf64 (_Float64 __x) throw (); extern _Float64 __sqrtf64 (_Float64 __x) throw ();



extern _Float64 hypotf64 (_Float64 __x, _Float64 __y) throw (); extern _Float64 __hypotf64 (_Float64 __x, _Float64 __y) throw ();




extern _Float64 cbrtf64 (_Float64 __x) throw (); extern _Float64 __cbrtf64 (_Float64 __x) throw ();






extern _Float64 ceilf64 (_Float64 __x) throw () __attribute__ ((__const__)); extern _Float64 __ceilf64 (_Float64 __x) throw () __attribute__ ((__const__));


extern _Float64 fabsf64 (_Float64 __x) throw () __attribute__ ((__const__)); extern _Float64 __fabsf64 (_Float64 __x) throw () __attribute__ ((__const__));


extern _Float64 floorf64 (_Float64 __x) throw () __attribute__ ((__const__)); extern _Float64 __floorf64 (_Float64 __x) throw () __attribute__ ((__const__));


extern _Float64 fmodf64 (_Float64 __x, _Float64 __y) throw (); extern _Float64 __fmodf64 (_Float64 __x, _Float64 __y) throw ();
# 198 "/usr/include/bits/mathcalls.h" 3 4
extern _Float64 copysignf64 (_Float64 __x, _Float64 __y) throw () __attribute__ ((__const__)); extern _Float64 __copysignf64 (_Float64 __x, _Float64 __y) throw () __attribute__ ((__const__));




extern _Float64 nanf64 (const char *__tagb) throw (); extern _Float64 __nanf64 (const char *__tagb) throw ();
# 220 "/usr/include/bits/mathcalls.h" 3 4
extern _Float64 j0f64 (_Float64) throw (); extern _Float64 __j0f64 (_Float64) throw ();
extern _Float64 j1f64 (_Float64) throw (); extern _Float64 __j1f64 (_Float64) throw ();
extern _Float64 jnf64 (int, _Float64) throw (); extern _Float64 __jnf64 (int, _Float64) throw ();
extern _Float64 y0f64 (_Float64) throw (); extern _Float64 __y0f64 (_Float64) throw ();
extern _Float64 y1f64 (_Float64) throw (); extern _Float64 __y1f64 (_Float64) throw ();
extern _Float64 ynf64 (int, _Float64) throw (); extern _Float64 __ynf64 (int, _Float64) throw ();





extern _Float64 erff64 (_Float64) throw (); extern _Float64 __erff64 (_Float64) throw ();
extern _Float64 erfcf64 (_Float64) throw (); extern _Float64 __erfcf64 (_Float64) throw ();
extern _Float64 lgammaf64 (_Float64) throw (); extern _Float64 __lgammaf64 (_Float64) throw ();




extern _Float64 tgammaf64 (_Float64) throw (); extern _Float64 __tgammaf64 (_Float64) throw ();
# 252 "/usr/include/bits/mathcalls.h" 3 4
extern _Float64 lgammaf64_r (_Float64, int *__signgamp) throw (); extern _Float64 __lgammaf64_r (_Float64, int *__signgamp) throw ();






extern _Float64 rintf64 (_Float64 __x) throw (); extern _Float64 __rintf64 (_Float64 __x) throw ();


extern _Float64 nextafterf64 (_Float64 __x, _Float64 __y) throw (); extern _Float64 __nextafterf64 (_Float64 __x, _Float64 __y) throw ();






extern _Float64 nextdownf64 (_Float64 __x) throw (); extern _Float64 __nextdownf64 (_Float64 __x) throw ();

extern _Float64 nextupf64 (_Float64 __x) throw (); extern _Float64 __nextupf64 (_Float64 __x) throw ();



extern _Float64 remainderf64 (_Float64 __x, _Float64 __y) throw (); extern _Float64 __remainderf64 (_Float64 __x, _Float64 __y) throw ();



extern _Float64 scalbnf64 (_Float64 __x, int __n) throw (); extern _Float64 __scalbnf64 (_Float64 __x, int __n) throw ();



extern int ilogbf64 (_Float64 __x) throw (); extern int __ilogbf64 (_Float64 __x) throw ();




extern long int llogbf64 (_Float64 __x) throw (); extern long int __llogbf64 (_Float64 __x) throw ();




extern _Float64 scalblnf64 (_Float64 __x, long int __n) throw (); extern _Float64 __scalblnf64 (_Float64 __x, long int __n) throw ();



extern _Float64 nearbyintf64 (_Float64 __x) throw (); extern _Float64 __nearbyintf64 (_Float64 __x) throw ();



extern _Float64 roundf64 (_Float64 __x) throw () __attribute__ ((__const__)); extern _Float64 __roundf64 (_Float64 __x) throw () __attribute__ ((__const__));



extern _Float64 truncf64 (_Float64 __x) throw () __attribute__ ((__const__)); extern _Float64 __truncf64 (_Float64 __x) throw () __attribute__ ((__const__));




extern _Float64 remquof64 (_Float64 __x, _Float64 __y, int *__quo) throw (); extern _Float64 __remquof64 (_Float64 __x, _Float64 __y, int *__quo) throw ();






extern long int lrintf64 (_Float64 __x) throw (); extern long int __lrintf64 (_Float64 __x) throw ();
__extension__
extern long long int llrintf64 (_Float64 __x) throw (); extern long long int __llrintf64 (_Float64 __x) throw ();



extern long int lroundf64 (_Float64 __x) throw (); extern long int __lroundf64 (_Float64 __x) throw ();
__extension__
extern long long int llroundf64 (_Float64 __x) throw (); extern long long int __llroundf64 (_Float64 __x) throw ();



extern _Float64 fdimf64 (_Float64 __x, _Float64 __y) throw (); extern _Float64 __fdimf64 (_Float64 __x, _Float64 __y) throw ();


extern _Float64 fmaxf64 (_Float64 __x, _Float64 __y) throw () __attribute__ ((__const__)); extern _Float64 __fmaxf64 (_Float64 __x, _Float64 __y) throw () __attribute__ ((__const__));


extern _Float64 fminf64 (_Float64 __x, _Float64 __y) throw () __attribute__ ((__const__)); extern _Float64 __fminf64 (_Float64 __x, _Float64 __y) throw () __attribute__ ((__const__));


extern _Float64 fmaf64 (_Float64 __x, _Float64 __y, _Float64 __z) throw (); extern _Float64 __fmaf64 (_Float64 __x, _Float64 __y, _Float64 __z) throw ();




extern _Float64 roundevenf64 (_Float64 __x) throw () __attribute__ ((__const__)); extern _Float64 __roundevenf64 (_Float64 __x) throw () __attribute__ ((__const__));



extern __intmax_t fromfpf64 (_Float64 __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpf64 (_Float64 __x, int __round, unsigned int __width) throw ()
                            ;



extern __uintmax_t ufromfpf64 (_Float64 __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpf64 (_Float64 __x, int __round, unsigned int __width) throw ()
                              ;




extern __intmax_t fromfpxf64 (_Float64 __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpxf64 (_Float64 __x, int __round, unsigned int __width) throw ()
                             ;




extern __uintmax_t ufromfpxf64 (_Float64 __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpxf64 (_Float64 __x, int __round, unsigned int __width) throw ()
                               ;


extern _Float64 fmaxmagf64 (_Float64 __x, _Float64 __y) throw () __attribute__ ((__const__)); extern _Float64 __fmaxmagf64 (_Float64 __x, _Float64 __y) throw () __attribute__ ((__const__));


extern _Float64 fminmagf64 (_Float64 __x, _Float64 __y) throw () __attribute__ ((__const__)); extern _Float64 __fminmagf64 (_Float64 __x, _Float64 __y) throw () __attribute__ ((__const__));


extern int canonicalizef64 (_Float64 *__cx, const _Float64 *__x) throw ();




extern int totalorderf64 (const _Float64 *__x, const _Float64 *__y) throw ()

     __attribute__ ((__pure__));


extern int totalordermagf64 (const _Float64 *__x, const _Float64 *__y) throw ()

     __attribute__ ((__pure__));


extern _Float64 getpayloadf64 (const _Float64 *__x) throw (); extern _Float64 __getpayloadf64 (const _Float64 *__x) throw ();


extern int setpayloadf64 (_Float64 *__x, _Float64 __payload) throw ();


extern int setpayloadsigf64 (_Float64 *__x, _Float64 __payload) throw ();
# 454 "/usr/include/math.h" 2 3 4
# 467 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-helper-functions.h" 1 3 4
# 20 "/usr/include/bits/mathcalls-helper-functions.h" 3 4
extern int __fpclassifyf128 (_Float128 __value) throw ()
     __attribute__ ((__const__));


extern int __signbitf128 (_Float128 __value) throw ()
     __attribute__ ((__const__));



extern int __isinff128 (_Float128 __value) throw ()
  __attribute__ ((__const__));


extern int __finitef128 (_Float128 __value) throw ()
  __attribute__ ((__const__));


extern int __isnanf128 (_Float128 __value) throw ()
  __attribute__ ((__const__));


extern int __iseqsigf128 (_Float128 __x, _Float128 __y) throw ();


extern int __issignalingf128 (_Float128 __value) throw ()
     __attribute__ ((__const__));
# 468 "/usr/include/math.h" 2 3 4


# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 53 "/usr/include/bits/mathcalls.h" 3 4
extern _Float128 acosf128 (_Float128 __x) throw (); extern _Float128 __acosf128 (_Float128 __x) throw ();

extern _Float128 asinf128 (_Float128 __x) throw (); extern _Float128 __asinf128 (_Float128 __x) throw ();

extern _Float128 atanf128 (_Float128 __x) throw (); extern _Float128 __atanf128 (_Float128 __x) throw ();

extern _Float128 atan2f128 (_Float128 __y, _Float128 __x) throw (); extern _Float128 __atan2f128 (_Float128 __y, _Float128 __x) throw ();


 extern _Float128 cosf128 (_Float128 __x) throw (); extern _Float128 __cosf128 (_Float128 __x) throw ();

 extern _Float128 sinf128 (_Float128 __x) throw (); extern _Float128 __sinf128 (_Float128 __x) throw ();

extern _Float128 tanf128 (_Float128 __x) throw (); extern _Float128 __tanf128 (_Float128 __x) throw ();




extern _Float128 coshf128 (_Float128 __x) throw (); extern _Float128 __coshf128 (_Float128 __x) throw ();

extern _Float128 sinhf128 (_Float128 __x) throw (); extern _Float128 __sinhf128 (_Float128 __x) throw ();

extern _Float128 tanhf128 (_Float128 __x) throw (); extern _Float128 __tanhf128 (_Float128 __x) throw ();



 extern void sincosf128 (_Float128 __x, _Float128 *__sinx, _Float128 *__cosx) throw (); extern void __sincosf128 (_Float128 __x, _Float128 *__sinx, _Float128 *__cosx) throw ()
                                                        ;




extern _Float128 acoshf128 (_Float128 __x) throw (); extern _Float128 __acoshf128 (_Float128 __x) throw ();

extern _Float128 asinhf128 (_Float128 __x) throw (); extern _Float128 __asinhf128 (_Float128 __x) throw ();

extern _Float128 atanhf128 (_Float128 __x) throw (); extern _Float128 __atanhf128 (_Float128 __x) throw ();





 extern _Float128 expf128 (_Float128 __x) throw (); extern _Float128 __expf128 (_Float128 __x) throw ();


extern _Float128 frexpf128 (_Float128 __x, int *__exponent) throw (); extern _Float128 __frexpf128 (_Float128 __x, int *__exponent) throw ();


extern _Float128 ldexpf128 (_Float128 __x, int __exponent) throw (); extern _Float128 __ldexpf128 (_Float128 __x, int __exponent) throw ();


 extern _Float128 logf128 (_Float128 __x) throw (); extern _Float128 __logf128 (_Float128 __x) throw ();


extern _Float128 log10f128 (_Float128 __x) throw (); extern _Float128 __log10f128 (_Float128 __x) throw ();


extern _Float128 modff128 (_Float128 __x, _Float128 *__iptr) throw (); extern _Float128 __modff128 (_Float128 __x, _Float128 *__iptr) throw () __attribute__ ((__nonnull__ (2)));



extern _Float128 exp10f128 (_Float128 __x) throw (); extern _Float128 __exp10f128 (_Float128 __x) throw ();




extern _Float128 expm1f128 (_Float128 __x) throw (); extern _Float128 __expm1f128 (_Float128 __x) throw ();


extern _Float128 log1pf128 (_Float128 __x) throw (); extern _Float128 __log1pf128 (_Float128 __x) throw ();


extern _Float128 logbf128 (_Float128 __x) throw (); extern _Float128 __logbf128 (_Float128 __x) throw ();




extern _Float128 exp2f128 (_Float128 __x) throw (); extern _Float128 __exp2f128 (_Float128 __x) throw ();


extern _Float128 log2f128 (_Float128 __x) throw (); extern _Float128 __log2f128 (_Float128 __x) throw ();






 extern _Float128 powf128 (_Float128 __x, _Float128 __y) throw (); extern _Float128 __powf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float128 sqrtf128 (_Float128 __x) throw (); extern _Float128 __sqrtf128 (_Float128 __x) throw ();



extern _Float128 hypotf128 (_Float128 __x, _Float128 __y) throw (); extern _Float128 __hypotf128 (_Float128 __x, _Float128 __y) throw ();




extern _Float128 cbrtf128 (_Float128 __x) throw (); extern _Float128 __cbrtf128 (_Float128 __x) throw ();






extern _Float128 ceilf128 (_Float128 __x) throw () __attribute__ ((__const__)); extern _Float128 __ceilf128 (_Float128 __x) throw () __attribute__ ((__const__));


extern _Float128 fabsf128 (_Float128 __x) throw () __attribute__ ((__const__)); extern _Float128 __fabsf128 (_Float128 __x) throw () __attribute__ ((__const__));


extern _Float128 floorf128 (_Float128 __x) throw () __attribute__ ((__const__)); extern _Float128 __floorf128 (_Float128 __x) throw () __attribute__ ((__const__));


extern _Float128 fmodf128 (_Float128 __x, _Float128 __y) throw (); extern _Float128 __fmodf128 (_Float128 __x, _Float128 __y) throw ();
# 198 "/usr/include/bits/mathcalls.h" 3 4
extern _Float128 copysignf128 (_Float128 __x, _Float128 __y) throw () __attribute__ ((__const__)); extern _Float128 __copysignf128 (_Float128 __x, _Float128 __y) throw () __attribute__ ((__const__));




extern _Float128 nanf128 (const char *__tagb) throw (); extern _Float128 __nanf128 (const char *__tagb) throw ();
# 220 "/usr/include/bits/mathcalls.h" 3 4
extern _Float128 j0f128 (_Float128) throw (); extern _Float128 __j0f128 (_Float128) throw ();
extern _Float128 j1f128 (_Float128) throw (); extern _Float128 __j1f128 (_Float128) throw ();
extern _Float128 jnf128 (int, _Float128) throw (); extern _Float128 __jnf128 (int, _Float128) throw ();
extern _Float128 y0f128 (_Float128) throw (); extern _Float128 __y0f128 (_Float128) throw ();
extern _Float128 y1f128 (_Float128) throw (); extern _Float128 __y1f128 (_Float128) throw ();
extern _Float128 ynf128 (int, _Float128) throw (); extern _Float128 __ynf128 (int, _Float128) throw ();





extern _Float128 erff128 (_Float128) throw (); extern _Float128 __erff128 (_Float128) throw ();
extern _Float128 erfcf128 (_Float128) throw (); extern _Float128 __erfcf128 (_Float128) throw ();
extern _Float128 lgammaf128 (_Float128) throw (); extern _Float128 __lgammaf128 (_Float128) throw ();




extern _Float128 tgammaf128 (_Float128) throw (); extern _Float128 __tgammaf128 (_Float128) throw ();
# 252 "/usr/include/bits/mathcalls.h" 3 4
extern _Float128 lgammaf128_r (_Float128, int *__signgamp) throw (); extern _Float128 __lgammaf128_r (_Float128, int *__signgamp) throw ();






extern _Float128 rintf128 (_Float128 __x) throw (); extern _Float128 __rintf128 (_Float128 __x) throw ();


extern _Float128 nextafterf128 (_Float128 __x, _Float128 __y) throw (); extern _Float128 __nextafterf128 (_Float128 __x, _Float128 __y) throw ();






extern _Float128 nextdownf128 (_Float128 __x) throw (); extern _Float128 __nextdownf128 (_Float128 __x) throw ();

extern _Float128 nextupf128 (_Float128 __x) throw (); extern _Float128 __nextupf128 (_Float128 __x) throw ();



extern _Float128 remainderf128 (_Float128 __x, _Float128 __y) throw (); extern _Float128 __remainderf128 (_Float128 __x, _Float128 __y) throw ();



extern _Float128 scalbnf128 (_Float128 __x, int __n) throw (); extern _Float128 __scalbnf128 (_Float128 __x, int __n) throw ();



extern int ilogbf128 (_Float128 __x) throw (); extern int __ilogbf128 (_Float128 __x) throw ();




extern long int llogbf128 (_Float128 __x) throw (); extern long int __llogbf128 (_Float128 __x) throw ();




extern _Float128 scalblnf128 (_Float128 __x, long int __n) throw (); extern _Float128 __scalblnf128 (_Float128 __x, long int __n) throw ();



extern _Float128 nearbyintf128 (_Float128 __x) throw (); extern _Float128 __nearbyintf128 (_Float128 __x) throw ();



extern _Float128 roundf128 (_Float128 __x) throw () __attribute__ ((__const__)); extern _Float128 __roundf128 (_Float128 __x) throw () __attribute__ ((__const__));



extern _Float128 truncf128 (_Float128 __x) throw () __attribute__ ((__const__)); extern _Float128 __truncf128 (_Float128 __x) throw () __attribute__ ((__const__));




extern _Float128 remquof128 (_Float128 __x, _Float128 __y, int *__quo) throw (); extern _Float128 __remquof128 (_Float128 __x, _Float128 __y, int *__quo) throw ();






extern long int lrintf128 (_Float128 __x) throw (); extern long int __lrintf128 (_Float128 __x) throw ();
__extension__
extern long long int llrintf128 (_Float128 __x) throw (); extern long long int __llrintf128 (_Float128 __x) throw ();



extern long int lroundf128 (_Float128 __x) throw (); extern long int __lroundf128 (_Float128 __x) throw ();
__extension__
extern long long int llroundf128 (_Float128 __x) throw (); extern long long int __llroundf128 (_Float128 __x) throw ();



extern _Float128 fdimf128 (_Float128 __x, _Float128 __y) throw (); extern _Float128 __fdimf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float128 fmaxf128 (_Float128 __x, _Float128 __y) throw () __attribute__ ((__const__)); extern _Float128 __fmaxf128 (_Float128 __x, _Float128 __y) throw () __attribute__ ((__const__));


extern _Float128 fminf128 (_Float128 __x, _Float128 __y) throw () __attribute__ ((__const__)); extern _Float128 __fminf128 (_Float128 __x, _Float128 __y) throw () __attribute__ ((__const__));


extern _Float128 fmaf128 (_Float128 __x, _Float128 __y, _Float128 __z) throw (); extern _Float128 __fmaf128 (_Float128 __x, _Float128 __y, _Float128 __z) throw ();




extern _Float128 roundevenf128 (_Float128 __x) throw () __attribute__ ((__const__)); extern _Float128 __roundevenf128 (_Float128 __x) throw () __attribute__ ((__const__));



extern __intmax_t fromfpf128 (_Float128 __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpf128 (_Float128 __x, int __round, unsigned int __width) throw ()
                            ;



extern __uintmax_t ufromfpf128 (_Float128 __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpf128 (_Float128 __x, int __round, unsigned int __width) throw ()
                              ;




extern __intmax_t fromfpxf128 (_Float128 __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpxf128 (_Float128 __x, int __round, unsigned int __width) throw ()
                             ;




extern __uintmax_t ufromfpxf128 (_Float128 __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpxf128 (_Float128 __x, int __round, unsigned int __width) throw ()
                               ;


extern _Float128 fmaxmagf128 (_Float128 __x, _Float128 __y) throw () __attribute__ ((__const__)); extern _Float128 __fmaxmagf128 (_Float128 __x, _Float128 __y) throw () __attribute__ ((__const__));


extern _Float128 fminmagf128 (_Float128 __x, _Float128 __y) throw () __attribute__ ((__const__)); extern _Float128 __fminmagf128 (_Float128 __x, _Float128 __y) throw () __attribute__ ((__const__));


extern int canonicalizef128 (_Float128 *__cx, const _Float128 *__x) throw ();




extern int totalorderf128 (const _Float128 *__x, const _Float128 *__y) throw ()

     __attribute__ ((__pure__));


extern int totalordermagf128 (const _Float128 *__x, const _Float128 *__y) throw ()

     __attribute__ ((__pure__));


extern _Float128 getpayloadf128 (const _Float128 *__x) throw (); extern _Float128 __getpayloadf128 (const _Float128 *__x) throw ();


extern int setpayloadf128 (_Float128 *__x, _Float128 __payload) throw ();


extern int setpayloadsigf128 (_Float128 *__x, _Float128 __payload) throw ();
# 471 "/usr/include/math.h" 2 3 4
# 487 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 53 "/usr/include/bits/mathcalls.h" 3 4
extern _Float32x acosf32x (_Float32x __x) throw (); extern _Float32x __acosf32x (_Float32x __x) throw ();

extern _Float32x asinf32x (_Float32x __x) throw (); extern _Float32x __asinf32x (_Float32x __x) throw ();

extern _Float32x atanf32x (_Float32x __x) throw (); extern _Float32x __atanf32x (_Float32x __x) throw ();

extern _Float32x atan2f32x (_Float32x __y, _Float32x __x) throw (); extern _Float32x __atan2f32x (_Float32x __y, _Float32x __x) throw ();


 extern _Float32x cosf32x (_Float32x __x) throw (); extern _Float32x __cosf32x (_Float32x __x) throw ();

 extern _Float32x sinf32x (_Float32x __x) throw (); extern _Float32x __sinf32x (_Float32x __x) throw ();

extern _Float32x tanf32x (_Float32x __x) throw (); extern _Float32x __tanf32x (_Float32x __x) throw ();




extern _Float32x coshf32x (_Float32x __x) throw (); extern _Float32x __coshf32x (_Float32x __x) throw ();

extern _Float32x sinhf32x (_Float32x __x) throw (); extern _Float32x __sinhf32x (_Float32x __x) throw ();

extern _Float32x tanhf32x (_Float32x __x) throw (); extern _Float32x __tanhf32x (_Float32x __x) throw ();



 extern void sincosf32x (_Float32x __x, _Float32x *__sinx, _Float32x *__cosx) throw (); extern void __sincosf32x (_Float32x __x, _Float32x *__sinx, _Float32x *__cosx) throw ()
                                                        ;




extern _Float32x acoshf32x (_Float32x __x) throw (); extern _Float32x __acoshf32x (_Float32x __x) throw ();

extern _Float32x asinhf32x (_Float32x __x) throw (); extern _Float32x __asinhf32x (_Float32x __x) throw ();

extern _Float32x atanhf32x (_Float32x __x) throw (); extern _Float32x __atanhf32x (_Float32x __x) throw ();





 extern _Float32x expf32x (_Float32x __x) throw (); extern _Float32x __expf32x (_Float32x __x) throw ();


extern _Float32x frexpf32x (_Float32x __x, int *__exponent) throw (); extern _Float32x __frexpf32x (_Float32x __x, int *__exponent) throw ();


extern _Float32x ldexpf32x (_Float32x __x, int __exponent) throw (); extern _Float32x __ldexpf32x (_Float32x __x, int __exponent) throw ();


 extern _Float32x logf32x (_Float32x __x) throw (); extern _Float32x __logf32x (_Float32x __x) throw ();


extern _Float32x log10f32x (_Float32x __x) throw (); extern _Float32x __log10f32x (_Float32x __x) throw ();


extern _Float32x modff32x (_Float32x __x, _Float32x *__iptr) throw (); extern _Float32x __modff32x (_Float32x __x, _Float32x *__iptr) throw () __attribute__ ((__nonnull__ (2)));



extern _Float32x exp10f32x (_Float32x __x) throw (); extern _Float32x __exp10f32x (_Float32x __x) throw ();




extern _Float32x expm1f32x (_Float32x __x) throw (); extern _Float32x __expm1f32x (_Float32x __x) throw ();


extern _Float32x log1pf32x (_Float32x __x) throw (); extern _Float32x __log1pf32x (_Float32x __x) throw ();


extern _Float32x logbf32x (_Float32x __x) throw (); extern _Float32x __logbf32x (_Float32x __x) throw ();




extern _Float32x exp2f32x (_Float32x __x) throw (); extern _Float32x __exp2f32x (_Float32x __x) throw ();


extern _Float32x log2f32x (_Float32x __x) throw (); extern _Float32x __log2f32x (_Float32x __x) throw ();






 extern _Float32x powf32x (_Float32x __x, _Float32x __y) throw (); extern _Float32x __powf32x (_Float32x __x, _Float32x __y) throw ();


extern _Float32x sqrtf32x (_Float32x __x) throw (); extern _Float32x __sqrtf32x (_Float32x __x) throw ();



extern _Float32x hypotf32x (_Float32x __x, _Float32x __y) throw (); extern _Float32x __hypotf32x (_Float32x __x, _Float32x __y) throw ();




extern _Float32x cbrtf32x (_Float32x __x) throw (); extern _Float32x __cbrtf32x (_Float32x __x) throw ();






extern _Float32x ceilf32x (_Float32x __x) throw () __attribute__ ((__const__)); extern _Float32x __ceilf32x (_Float32x __x) throw () __attribute__ ((__const__));


extern _Float32x fabsf32x (_Float32x __x) throw () __attribute__ ((__const__)); extern _Float32x __fabsf32x (_Float32x __x) throw () __attribute__ ((__const__));


extern _Float32x floorf32x (_Float32x __x) throw () __attribute__ ((__const__)); extern _Float32x __floorf32x (_Float32x __x) throw () __attribute__ ((__const__));


extern _Float32x fmodf32x (_Float32x __x, _Float32x __y) throw (); extern _Float32x __fmodf32x (_Float32x __x, _Float32x __y) throw ();
# 198 "/usr/include/bits/mathcalls.h" 3 4
extern _Float32x copysignf32x (_Float32x __x, _Float32x __y) throw () __attribute__ ((__const__)); extern _Float32x __copysignf32x (_Float32x __x, _Float32x __y) throw () __attribute__ ((__const__));




extern _Float32x nanf32x (const char *__tagb) throw (); extern _Float32x __nanf32x (const char *__tagb) throw ();
# 220 "/usr/include/bits/mathcalls.h" 3 4
extern _Float32x j0f32x (_Float32x) throw (); extern _Float32x __j0f32x (_Float32x) throw ();
extern _Float32x j1f32x (_Float32x) throw (); extern _Float32x __j1f32x (_Float32x) throw ();
extern _Float32x jnf32x (int, _Float32x) throw (); extern _Float32x __jnf32x (int, _Float32x) throw ();
extern _Float32x y0f32x (_Float32x) throw (); extern _Float32x __y0f32x (_Float32x) throw ();
extern _Float32x y1f32x (_Float32x) throw (); extern _Float32x __y1f32x (_Float32x) throw ();
extern _Float32x ynf32x (int, _Float32x) throw (); extern _Float32x __ynf32x (int, _Float32x) throw ();





extern _Float32x erff32x (_Float32x) throw (); extern _Float32x __erff32x (_Float32x) throw ();
extern _Float32x erfcf32x (_Float32x) throw (); extern _Float32x __erfcf32x (_Float32x) throw ();
extern _Float32x lgammaf32x (_Float32x) throw (); extern _Float32x __lgammaf32x (_Float32x) throw ();




extern _Float32x tgammaf32x (_Float32x) throw (); extern _Float32x __tgammaf32x (_Float32x) throw ();
# 252 "/usr/include/bits/mathcalls.h" 3 4
extern _Float32x lgammaf32x_r (_Float32x, int *__signgamp) throw (); extern _Float32x __lgammaf32x_r (_Float32x, int *__signgamp) throw ();






extern _Float32x rintf32x (_Float32x __x) throw (); extern _Float32x __rintf32x (_Float32x __x) throw ();


extern _Float32x nextafterf32x (_Float32x __x, _Float32x __y) throw (); extern _Float32x __nextafterf32x (_Float32x __x, _Float32x __y) throw ();






extern _Float32x nextdownf32x (_Float32x __x) throw (); extern _Float32x __nextdownf32x (_Float32x __x) throw ();

extern _Float32x nextupf32x (_Float32x __x) throw (); extern _Float32x __nextupf32x (_Float32x __x) throw ();



extern _Float32x remainderf32x (_Float32x __x, _Float32x __y) throw (); extern _Float32x __remainderf32x (_Float32x __x, _Float32x __y) throw ();



extern _Float32x scalbnf32x (_Float32x __x, int __n) throw (); extern _Float32x __scalbnf32x (_Float32x __x, int __n) throw ();



extern int ilogbf32x (_Float32x __x) throw (); extern int __ilogbf32x (_Float32x __x) throw ();




extern long int llogbf32x (_Float32x __x) throw (); extern long int __llogbf32x (_Float32x __x) throw ();




extern _Float32x scalblnf32x (_Float32x __x, long int __n) throw (); extern _Float32x __scalblnf32x (_Float32x __x, long int __n) throw ();



extern _Float32x nearbyintf32x (_Float32x __x) throw (); extern _Float32x __nearbyintf32x (_Float32x __x) throw ();



extern _Float32x roundf32x (_Float32x __x) throw () __attribute__ ((__const__)); extern _Float32x __roundf32x (_Float32x __x) throw () __attribute__ ((__const__));



extern _Float32x truncf32x (_Float32x __x) throw () __attribute__ ((__const__)); extern _Float32x __truncf32x (_Float32x __x) throw () __attribute__ ((__const__));




extern _Float32x remquof32x (_Float32x __x, _Float32x __y, int *__quo) throw (); extern _Float32x __remquof32x (_Float32x __x, _Float32x __y, int *__quo) throw ();






extern long int lrintf32x (_Float32x __x) throw (); extern long int __lrintf32x (_Float32x __x) throw ();
__extension__
extern long long int llrintf32x (_Float32x __x) throw (); extern long long int __llrintf32x (_Float32x __x) throw ();



extern long int lroundf32x (_Float32x __x) throw (); extern long int __lroundf32x (_Float32x __x) throw ();
__extension__
extern long long int llroundf32x (_Float32x __x) throw (); extern long long int __llroundf32x (_Float32x __x) throw ();



extern _Float32x fdimf32x (_Float32x __x, _Float32x __y) throw (); extern _Float32x __fdimf32x (_Float32x __x, _Float32x __y) throw ();


extern _Float32x fmaxf32x (_Float32x __x, _Float32x __y) throw () __attribute__ ((__const__)); extern _Float32x __fmaxf32x (_Float32x __x, _Float32x __y) throw () __attribute__ ((__const__));


extern _Float32x fminf32x (_Float32x __x, _Float32x __y) throw () __attribute__ ((__const__)); extern _Float32x __fminf32x (_Float32x __x, _Float32x __y) throw () __attribute__ ((__const__));


extern _Float32x fmaf32x (_Float32x __x, _Float32x __y, _Float32x __z) throw (); extern _Float32x __fmaf32x (_Float32x __x, _Float32x __y, _Float32x __z) throw ();




extern _Float32x roundevenf32x (_Float32x __x) throw () __attribute__ ((__const__)); extern _Float32x __roundevenf32x (_Float32x __x) throw () __attribute__ ((__const__));



extern __intmax_t fromfpf32x (_Float32x __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpf32x (_Float32x __x, int __round, unsigned int __width) throw ()
                            ;



extern __uintmax_t ufromfpf32x (_Float32x __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpf32x (_Float32x __x, int __round, unsigned int __width) throw ()
                              ;




extern __intmax_t fromfpxf32x (_Float32x __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpxf32x (_Float32x __x, int __round, unsigned int __width) throw ()
                             ;




extern __uintmax_t ufromfpxf32x (_Float32x __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpxf32x (_Float32x __x, int __round, unsigned int __width) throw ()
                               ;


extern _Float32x fmaxmagf32x (_Float32x __x, _Float32x __y) throw () __attribute__ ((__const__)); extern _Float32x __fmaxmagf32x (_Float32x __x, _Float32x __y) throw () __attribute__ ((__const__));


extern _Float32x fminmagf32x (_Float32x __x, _Float32x __y) throw () __attribute__ ((__const__)); extern _Float32x __fminmagf32x (_Float32x __x, _Float32x __y) throw () __attribute__ ((__const__));


extern int canonicalizef32x (_Float32x *__cx, const _Float32x *__x) throw ();




extern int totalorderf32x (const _Float32x *__x, const _Float32x *__y) throw ()

     __attribute__ ((__pure__));


extern int totalordermagf32x (const _Float32x *__x, const _Float32x *__y) throw ()

     __attribute__ ((__pure__));


extern _Float32x getpayloadf32x (const _Float32x *__x) throw (); extern _Float32x __getpayloadf32x (const _Float32x *__x) throw ();


extern int setpayloadf32x (_Float32x *__x, _Float32x __payload) throw ();


extern int setpayloadsigf32x (_Float32x *__x, _Float32x __payload) throw ();
# 488 "/usr/include/math.h" 2 3 4
# 504 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 53 "/usr/include/bits/mathcalls.h" 3 4
extern _Float64x acosf64x (_Float64x __x) throw (); extern _Float64x __acosf64x (_Float64x __x) throw ();

extern _Float64x asinf64x (_Float64x __x) throw (); extern _Float64x __asinf64x (_Float64x __x) throw ();

extern _Float64x atanf64x (_Float64x __x) throw (); extern _Float64x __atanf64x (_Float64x __x) throw ();

extern _Float64x atan2f64x (_Float64x __y, _Float64x __x) throw (); extern _Float64x __atan2f64x (_Float64x __y, _Float64x __x) throw ();


 extern _Float64x cosf64x (_Float64x __x) throw (); extern _Float64x __cosf64x (_Float64x __x) throw ();

 extern _Float64x sinf64x (_Float64x __x) throw (); extern _Float64x __sinf64x (_Float64x __x) throw ();

extern _Float64x tanf64x (_Float64x __x) throw (); extern _Float64x __tanf64x (_Float64x __x) throw ();




extern _Float64x coshf64x (_Float64x __x) throw (); extern _Float64x __coshf64x (_Float64x __x) throw ();

extern _Float64x sinhf64x (_Float64x __x) throw (); extern _Float64x __sinhf64x (_Float64x __x) throw ();

extern _Float64x tanhf64x (_Float64x __x) throw (); extern _Float64x __tanhf64x (_Float64x __x) throw ();



 extern void sincosf64x (_Float64x __x, _Float64x *__sinx, _Float64x *__cosx) throw (); extern void __sincosf64x (_Float64x __x, _Float64x *__sinx, _Float64x *__cosx) throw ()
                                                        ;




extern _Float64x acoshf64x (_Float64x __x) throw (); extern _Float64x __acoshf64x (_Float64x __x) throw ();

extern _Float64x asinhf64x (_Float64x __x) throw (); extern _Float64x __asinhf64x (_Float64x __x) throw ();

extern _Float64x atanhf64x (_Float64x __x) throw (); extern _Float64x __atanhf64x (_Float64x __x) throw ();





 extern _Float64x expf64x (_Float64x __x) throw (); extern _Float64x __expf64x (_Float64x __x) throw ();


extern _Float64x frexpf64x (_Float64x __x, int *__exponent) throw (); extern _Float64x __frexpf64x (_Float64x __x, int *__exponent) throw ();


extern _Float64x ldexpf64x (_Float64x __x, int __exponent) throw (); extern _Float64x __ldexpf64x (_Float64x __x, int __exponent) throw ();


 extern _Float64x logf64x (_Float64x __x) throw (); extern _Float64x __logf64x (_Float64x __x) throw ();


extern _Float64x log10f64x (_Float64x __x) throw (); extern _Float64x __log10f64x (_Float64x __x) throw ();


extern _Float64x modff64x (_Float64x __x, _Float64x *__iptr) throw (); extern _Float64x __modff64x (_Float64x __x, _Float64x *__iptr) throw () __attribute__ ((__nonnull__ (2)));



extern _Float64x exp10f64x (_Float64x __x) throw (); extern _Float64x __exp10f64x (_Float64x __x) throw ();




extern _Float64x expm1f64x (_Float64x __x) throw (); extern _Float64x __expm1f64x (_Float64x __x) throw ();


extern _Float64x log1pf64x (_Float64x __x) throw (); extern _Float64x __log1pf64x (_Float64x __x) throw ();


extern _Float64x logbf64x (_Float64x __x) throw (); extern _Float64x __logbf64x (_Float64x __x) throw ();




extern _Float64x exp2f64x (_Float64x __x) throw (); extern _Float64x __exp2f64x (_Float64x __x) throw ();


extern _Float64x log2f64x (_Float64x __x) throw (); extern _Float64x __log2f64x (_Float64x __x) throw ();






 extern _Float64x powf64x (_Float64x __x, _Float64x __y) throw (); extern _Float64x __powf64x (_Float64x __x, _Float64x __y) throw ();


extern _Float64x sqrtf64x (_Float64x __x) throw (); extern _Float64x __sqrtf64x (_Float64x __x) throw ();



extern _Float64x hypotf64x (_Float64x __x, _Float64x __y) throw (); extern _Float64x __hypotf64x (_Float64x __x, _Float64x __y) throw ();




extern _Float64x cbrtf64x (_Float64x __x) throw (); extern _Float64x __cbrtf64x (_Float64x __x) throw ();






extern _Float64x ceilf64x (_Float64x __x) throw () __attribute__ ((__const__)); extern _Float64x __ceilf64x (_Float64x __x) throw () __attribute__ ((__const__));


extern _Float64x fabsf64x (_Float64x __x) throw () __attribute__ ((__const__)); extern _Float64x __fabsf64x (_Float64x __x) throw () __attribute__ ((__const__));


extern _Float64x floorf64x (_Float64x __x) throw () __attribute__ ((__const__)); extern _Float64x __floorf64x (_Float64x __x) throw () __attribute__ ((__const__));


extern _Float64x fmodf64x (_Float64x __x, _Float64x __y) throw (); extern _Float64x __fmodf64x (_Float64x __x, _Float64x __y) throw ();
# 198 "/usr/include/bits/mathcalls.h" 3 4
extern _Float64x copysignf64x (_Float64x __x, _Float64x __y) throw () __attribute__ ((__const__)); extern _Float64x __copysignf64x (_Float64x __x, _Float64x __y) throw () __attribute__ ((__const__));




extern _Float64x nanf64x (const char *__tagb) throw (); extern _Float64x __nanf64x (const char *__tagb) throw ();
# 220 "/usr/include/bits/mathcalls.h" 3 4
extern _Float64x j0f64x (_Float64x) throw (); extern _Float64x __j0f64x (_Float64x) throw ();
extern _Float64x j1f64x (_Float64x) throw (); extern _Float64x __j1f64x (_Float64x) throw ();
extern _Float64x jnf64x (int, _Float64x) throw (); extern _Float64x __jnf64x (int, _Float64x) throw ();
extern _Float64x y0f64x (_Float64x) throw (); extern _Float64x __y0f64x (_Float64x) throw ();
extern _Float64x y1f64x (_Float64x) throw (); extern _Float64x __y1f64x (_Float64x) throw ();
extern _Float64x ynf64x (int, _Float64x) throw (); extern _Float64x __ynf64x (int, _Float64x) throw ();





extern _Float64x erff64x (_Float64x) throw (); extern _Float64x __erff64x (_Float64x) throw ();
extern _Float64x erfcf64x (_Float64x) throw (); extern _Float64x __erfcf64x (_Float64x) throw ();
extern _Float64x lgammaf64x (_Float64x) throw (); extern _Float64x __lgammaf64x (_Float64x) throw ();




extern _Float64x tgammaf64x (_Float64x) throw (); extern _Float64x __tgammaf64x (_Float64x) throw ();
# 252 "/usr/include/bits/mathcalls.h" 3 4
extern _Float64x lgammaf64x_r (_Float64x, int *__signgamp) throw (); extern _Float64x __lgammaf64x_r (_Float64x, int *__signgamp) throw ();






extern _Float64x rintf64x (_Float64x __x) throw (); extern _Float64x __rintf64x (_Float64x __x) throw ();


extern _Float64x nextafterf64x (_Float64x __x, _Float64x __y) throw (); extern _Float64x __nextafterf64x (_Float64x __x, _Float64x __y) throw ();






extern _Float64x nextdownf64x (_Float64x __x) throw (); extern _Float64x __nextdownf64x (_Float64x __x) throw ();

extern _Float64x nextupf64x (_Float64x __x) throw (); extern _Float64x __nextupf64x (_Float64x __x) throw ();



extern _Float64x remainderf64x (_Float64x __x, _Float64x __y) throw (); extern _Float64x __remainderf64x (_Float64x __x, _Float64x __y) throw ();



extern _Float64x scalbnf64x (_Float64x __x, int __n) throw (); extern _Float64x __scalbnf64x (_Float64x __x, int __n) throw ();



extern int ilogbf64x (_Float64x __x) throw (); extern int __ilogbf64x (_Float64x __x) throw ();




extern long int llogbf64x (_Float64x __x) throw (); extern long int __llogbf64x (_Float64x __x) throw ();




extern _Float64x scalblnf64x (_Float64x __x, long int __n) throw (); extern _Float64x __scalblnf64x (_Float64x __x, long int __n) throw ();



extern _Float64x nearbyintf64x (_Float64x __x) throw (); extern _Float64x __nearbyintf64x (_Float64x __x) throw ();



extern _Float64x roundf64x (_Float64x __x) throw () __attribute__ ((__const__)); extern _Float64x __roundf64x (_Float64x __x) throw () __attribute__ ((__const__));



extern _Float64x truncf64x (_Float64x __x) throw () __attribute__ ((__const__)); extern _Float64x __truncf64x (_Float64x __x) throw () __attribute__ ((__const__));




extern _Float64x remquof64x (_Float64x __x, _Float64x __y, int *__quo) throw (); extern _Float64x __remquof64x (_Float64x __x, _Float64x __y, int *__quo) throw ();






extern long int lrintf64x (_Float64x __x) throw (); extern long int __lrintf64x (_Float64x __x) throw ();
__extension__
extern long long int llrintf64x (_Float64x __x) throw (); extern long long int __llrintf64x (_Float64x __x) throw ();



extern long int lroundf64x (_Float64x __x) throw (); extern long int __lroundf64x (_Float64x __x) throw ();
__extension__
extern long long int llroundf64x (_Float64x __x) throw (); extern long long int __llroundf64x (_Float64x __x) throw ();



extern _Float64x fdimf64x (_Float64x __x, _Float64x __y) throw (); extern _Float64x __fdimf64x (_Float64x __x, _Float64x __y) throw ();


extern _Float64x fmaxf64x (_Float64x __x, _Float64x __y) throw () __attribute__ ((__const__)); extern _Float64x __fmaxf64x (_Float64x __x, _Float64x __y) throw () __attribute__ ((__const__));


extern _Float64x fminf64x (_Float64x __x, _Float64x __y) throw () __attribute__ ((__const__)); extern _Float64x __fminf64x (_Float64x __x, _Float64x __y) throw () __attribute__ ((__const__));


extern _Float64x fmaf64x (_Float64x __x, _Float64x __y, _Float64x __z) throw (); extern _Float64x __fmaf64x (_Float64x __x, _Float64x __y, _Float64x __z) throw ();




extern _Float64x roundevenf64x (_Float64x __x) throw () __attribute__ ((__const__)); extern _Float64x __roundevenf64x (_Float64x __x) throw () __attribute__ ((__const__));



extern __intmax_t fromfpf64x (_Float64x __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpf64x (_Float64x __x, int __round, unsigned int __width) throw ()
                            ;



extern __uintmax_t ufromfpf64x (_Float64x __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpf64x (_Float64x __x, int __round, unsigned int __width) throw ()
                              ;




extern __intmax_t fromfpxf64x (_Float64x __x, int __round, unsigned int __width) throw (); extern __intmax_t __fromfpxf64x (_Float64x __x, int __round, unsigned int __width) throw ()
                             ;




extern __uintmax_t ufromfpxf64x (_Float64x __x, int __round, unsigned int __width) throw (); extern __uintmax_t __ufromfpxf64x (_Float64x __x, int __round, unsigned int __width) throw ()
                               ;


extern _Float64x fmaxmagf64x (_Float64x __x, _Float64x __y) throw () __attribute__ ((__const__)); extern _Float64x __fmaxmagf64x (_Float64x __x, _Float64x __y) throw () __attribute__ ((__const__));


extern _Float64x fminmagf64x (_Float64x __x, _Float64x __y) throw () __attribute__ ((__const__)); extern _Float64x __fminmagf64x (_Float64x __x, _Float64x __y) throw () __attribute__ ((__const__));


extern int canonicalizef64x (_Float64x *__cx, const _Float64x *__x) throw ();




extern int totalorderf64x (const _Float64x *__x, const _Float64x *__y) throw ()

     __attribute__ ((__pure__));


extern int totalordermagf64x (const _Float64x *__x, const _Float64x *__y) throw ()

     __attribute__ ((__pure__));


extern _Float64x getpayloadf64x (const _Float64x *__x) throw (); extern _Float64x __getpayloadf64x (const _Float64x *__x) throw ();


extern int setpayloadf64x (_Float64x *__x, _Float64x __payload) throw ();


extern int setpayloadsigf64x (_Float64x *__x, _Float64x __payload) throw ();
# 505 "/usr/include/math.h" 2 3 4
# 552 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern float fadd (double __x, double __y) throw ();


extern float fdiv (double __x, double __y) throw ();


extern float fmul (double __x, double __y) throw ();


extern float fsub (double __x, double __y) throw ();
# 553 "/usr/include/math.h" 2 3 4
# 571 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern float faddl (long double __x, long double __y) throw ();


extern float fdivl (long double __x, long double __y) throw ();


extern float fmull (long double __x, long double __y) throw ();


extern float fsubl (long double __x, long double __y) throw ();
# 572 "/usr/include/math.h" 2 3 4
# 597 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern double daddl (long double __x, long double __y) throw ();


extern double ddivl (long double __x, long double __y) throw ();


extern double dmull (long double __x, long double __y) throw ();


extern double dsubl (long double __x, long double __y) throw ();
# 598 "/usr/include/math.h" 2 3 4
# 677 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern _Float32 f32addf32x (_Float32x __x, _Float32x __y) throw ();


extern _Float32 f32divf32x (_Float32x __x, _Float32x __y) throw ();


extern _Float32 f32mulf32x (_Float32x __x, _Float32x __y) throw ();


extern _Float32 f32subf32x (_Float32x __x, _Float32x __y) throw ();
# 678 "/usr/include/math.h" 2 3 4
# 687 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern _Float32 f32addf64 (_Float64 __x, _Float64 __y) throw ();


extern _Float32 f32divf64 (_Float64 __x, _Float64 __y) throw ();


extern _Float32 f32mulf64 (_Float64 __x, _Float64 __y) throw ();


extern _Float32 f32subf64 (_Float64 __x, _Float64 __y) throw ();
# 688 "/usr/include/math.h" 2 3 4
# 697 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern _Float32 f32addf64x (_Float64x __x, _Float64x __y) throw ();


extern _Float32 f32divf64x (_Float64x __x, _Float64x __y) throw ();


extern _Float32 f32mulf64x (_Float64x __x, _Float64x __y) throw ();


extern _Float32 f32subf64x (_Float64x __x, _Float64x __y) throw ();
# 698 "/usr/include/math.h" 2 3 4
# 707 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern _Float32 f32addf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float32 f32divf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float32 f32mulf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float32 f32subf128 (_Float128 __x, _Float128 __y) throw ();
# 708 "/usr/include/math.h" 2 3 4
# 727 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern _Float32x f32xaddf64 (_Float64 __x, _Float64 __y) throw ();


extern _Float32x f32xdivf64 (_Float64 __x, _Float64 __y) throw ();


extern _Float32x f32xmulf64 (_Float64 __x, _Float64 __y) throw ();


extern _Float32x f32xsubf64 (_Float64 __x, _Float64 __y) throw ();
# 728 "/usr/include/math.h" 2 3 4
# 737 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern _Float32x f32xaddf64x (_Float64x __x, _Float64x __y) throw ();


extern _Float32x f32xdivf64x (_Float64x __x, _Float64x __y) throw ();


extern _Float32x f32xmulf64x (_Float64x __x, _Float64x __y) throw ();


extern _Float32x f32xsubf64x (_Float64x __x, _Float64x __y) throw ();
# 738 "/usr/include/math.h" 2 3 4
# 747 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern _Float32x f32xaddf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float32x f32xdivf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float32x f32xmulf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float32x f32xsubf128 (_Float128 __x, _Float128 __y) throw ();
# 748 "/usr/include/math.h" 2 3 4
# 767 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern _Float64 f64addf64x (_Float64x __x, _Float64x __y) throw ();


extern _Float64 f64divf64x (_Float64x __x, _Float64x __y) throw ();


extern _Float64 f64mulf64x (_Float64x __x, _Float64x __y) throw ();


extern _Float64 f64subf64x (_Float64x __x, _Float64x __y) throw ();
# 768 "/usr/include/math.h" 2 3 4
# 777 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern _Float64 f64addf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float64 f64divf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float64 f64mulf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float64 f64subf128 (_Float128 __x, _Float128 __y) throw ();
# 778 "/usr/include/math.h" 2 3 4
# 797 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-narrow.h" 1 3 4
# 24 "/usr/include/bits/mathcalls-narrow.h" 3 4
extern _Float64x f64xaddf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float64x f64xdivf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float64x f64xmulf128 (_Float128 __x, _Float128 __y) throw ();


extern _Float64x f64xsubf128 (_Float128 __x, _Float128 __y) throw ();
# 798 "/usr/include/math.h" 2 3 4
# 834 "/usr/include/math.h" 3 4
extern int signgam;
# 914 "/usr/include/math.h" 3 4
enum
  {
    FP_NAN =

      0,
    FP_INFINITE =

      1,
    FP_ZERO =

      2,
    FP_SUBNORMAL =

      3,
    FP_NORMAL =

      4
  };
# 1034 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/iscanonical.h" 1 3 4
# 23 "/usr/include/bits/iscanonical.h" 3 4
extern int __iscanonicall (long double __x)
     throw () __attribute__ ((__const__));
# 46 "/usr/include/bits/iscanonical.h" 3 4
extern "C++" {
inline int iscanonical (float __val) { return ((void) (__typeof (__val)) (__val), 1); }
inline int iscanonical (double __val) { return ((void) (__typeof (__val)) (__val), 1); }
inline int iscanonical (long double __val) { return __iscanonicall (__val); }

inline int iscanonical (_Float128 __val) { return ((void) (__typeof (__val)) (__val), 1); }

}
# 1035 "/usr/include/math.h" 2 3 4
# 1046 "/usr/include/math.h" 3 4
extern "C++" {
inline int issignaling (float __val) { return __issignalingf (__val); }
inline int issignaling (double __val) { return __issignaling (__val); }
inline int
issignaling (long double __val)
{



  return __issignalingl (__val);

}



inline int issignaling (_Float128 __val) { return __issignalingf128 (__val); }

}
# 1077 "/usr/include/math.h" 3 4
extern "C++" {
# 1108 "/usr/include/math.h" 3 4
template <class __T> inline bool
iszero (__T __val)
{
  return __val == 0;
}

}
# 1326 "/usr/include/math.h" 3 4
extern "C++" {
template<typename> struct __iseqsig_type;

template<> struct __iseqsig_type<float>
{
  static int __call (float __x, float __y) throw ()
  {
    return __iseqsigf (__x, __y);
  }
};

template<> struct __iseqsig_type<double>
{
  static int __call (double __x, double __y) throw ()
  {
    return __iseqsig (__x, __y);
  }
};

template<> struct __iseqsig_type<long double>
{
  static int __call (long double __x, long double __y) throw ()
  {

    return __iseqsigl (__x, __y);



  }
};




template<> struct __iseqsig_type<_Float128>
{
  static int __call (_Float128 __x, _Float128 __y) throw ()
  {
    return __iseqsigf128 (__x, __y);
  }
};


template<typename _T1, typename _T2>
inline int
iseqsig (_T1 __x, _T2 __y) throw ()
{

  typedef decltype (((__x) + (__y) + 0.0f)) _T3;



  return __iseqsig_type<_T3>::__call (__x, __y);
}

}




}
# 46 "/usr/include/c++/10/cmath" 2 3
# 77 "/usr/include/c++/10/cmath" 3
extern "C++"
{
namespace std __attribute__ ((__visibility__ ("default")))
{


  using ::acos;


  inline constexpr float
  acos(float __x)
  { return __builtin_acosf(__x); }

  inline constexpr long double
  acos(long double __x)
  { return __builtin_acosl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    acos(_Tp __x)
    { return __builtin_acos(__x); }

  using ::asin;


  inline constexpr float
  asin(float __x)
  { return __builtin_asinf(__x); }

  inline constexpr long double
  asin(long double __x)
  { return __builtin_asinl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    asin(_Tp __x)
    { return __builtin_asin(__x); }

  using ::atan;


  inline constexpr float
  atan(float __x)
  { return __builtin_atanf(__x); }

  inline constexpr long double
  atan(long double __x)
  { return __builtin_atanl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    atan(_Tp __x)
    { return __builtin_atan(__x); }

  using ::atan2;


  inline constexpr float
  atan2(float __y, float __x)
  { return __builtin_atan2f(__y, __x); }

  inline constexpr long double
  atan2(long double __y, long double __x)
  { return __builtin_atan2l(__y, __x); }


  template<typename _Tp, typename _Up>
    inline constexpr
    typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    atan2(_Tp __y, _Up __x)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return atan2(__type(__y), __type(__x));
    }

  using ::ceil;


  inline constexpr float
  ceil(float __x)
  { return __builtin_ceilf(__x); }

  inline constexpr long double
  ceil(long double __x)
  { return __builtin_ceill(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    ceil(_Tp __x)
    { return __builtin_ceil(__x); }

  using ::cos;


  inline constexpr float
  cos(float __x)
  { return __builtin_cosf(__x); }

  inline constexpr long double
  cos(long double __x)
  { return __builtin_cosl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    cos(_Tp __x)
    { return __builtin_cos(__x); }

  using ::cosh;


  inline constexpr float
  cosh(float __x)
  { return __builtin_coshf(__x); }

  inline constexpr long double
  cosh(long double __x)
  { return __builtin_coshl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    cosh(_Tp __x)
    { return __builtin_cosh(__x); }

  using ::exp;


  inline constexpr float
  exp(float __x)
  { return __builtin_expf(__x); }

  inline constexpr long double
  exp(long double __x)
  { return __builtin_expl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    exp(_Tp __x)
    { return __builtin_exp(__x); }

  using ::fabs;


  inline constexpr float
  fabs(float __x)
  { return __builtin_fabsf(__x); }

  inline constexpr long double
  fabs(long double __x)
  { return __builtin_fabsl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    fabs(_Tp __x)
    { return __builtin_fabs(__x); }

  using ::floor;


  inline constexpr float
  floor(float __x)
  { return __builtin_floorf(__x); }

  inline constexpr long double
  floor(long double __x)
  { return __builtin_floorl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    floor(_Tp __x)
    { return __builtin_floor(__x); }

  using ::fmod;


  inline constexpr float
  fmod(float __x, float __y)
  { return __builtin_fmodf(__x, __y); }

  inline constexpr long double
  fmod(long double __x, long double __y)
  { return __builtin_fmodl(__x, __y); }


  template<typename _Tp, typename _Up>
    inline constexpr
    typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    fmod(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return fmod(__type(__x), __type(__y));
    }

  using ::frexp;


  inline float
  frexp(float __x, int* __exp)
  { return __builtin_frexpf(__x, __exp); }

  inline long double
  frexp(long double __x, int* __exp)
  { return __builtin_frexpl(__x, __exp); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    frexp(_Tp __x, int* __exp)
    { return __builtin_frexp(__x, __exp); }

  using ::ldexp;


  inline constexpr float
  ldexp(float __x, int __exp)
  { return __builtin_ldexpf(__x, __exp); }

  inline constexpr long double
  ldexp(long double __x, int __exp)
  { return __builtin_ldexpl(__x, __exp); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    ldexp(_Tp __x, int __exp)
    { return __builtin_ldexp(__x, __exp); }

  using ::log;


  inline constexpr float
  log(float __x)
  { return __builtin_logf(__x); }

  inline constexpr long double
  log(long double __x)
  { return __builtin_logl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    log(_Tp __x)
    { return __builtin_log(__x); }

  using ::log10;


  inline constexpr float
  log10(float __x)
  { return __builtin_log10f(__x); }

  inline constexpr long double
  log10(long double __x)
  { return __builtin_log10l(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    log10(_Tp __x)
    { return __builtin_log10(__x); }

  using ::modf;


  inline float
  modf(float __x, float* __iptr)
  { return __builtin_modff(__x, __iptr); }

  inline long double
  modf(long double __x, long double* __iptr)
  { return __builtin_modfl(__x, __iptr); }


  using ::pow;


  inline constexpr float
  pow(float __x, float __y)
  { return __builtin_powf(__x, __y); }

  inline constexpr long double
  pow(long double __x, long double __y)
  { return __builtin_powl(__x, __y); }
# 412 "/usr/include/c++/10/cmath" 3
  template<typename _Tp, typename _Up>
    inline constexpr
    typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    pow(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return pow(__type(__x), __type(__y));
    }

  using ::sin;


  inline constexpr float
  sin(float __x)
  { return __builtin_sinf(__x); }

  inline constexpr long double
  sin(long double __x)
  { return __builtin_sinl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    sin(_Tp __x)
    { return __builtin_sin(__x); }

  using ::sinh;


  inline constexpr float
  sinh(float __x)
  { return __builtin_sinhf(__x); }

  inline constexpr long double
  sinh(long double __x)
  { return __builtin_sinhl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    sinh(_Tp __x)
    { return __builtin_sinh(__x); }

  using ::sqrt;


  inline constexpr float
  sqrt(float __x)
  { return __builtin_sqrtf(__x); }

  inline constexpr long double
  sqrt(long double __x)
  { return __builtin_sqrtl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    sqrt(_Tp __x)
    { return __builtin_sqrt(__x); }

  using ::tan;


  inline constexpr float
  tan(float __x)
  { return __builtin_tanf(__x); }

  inline constexpr long double
  tan(long double __x)
  { return __builtin_tanl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    tan(_Tp __x)
    { return __builtin_tan(__x); }

  using ::tanh;


  inline constexpr float
  tanh(float __x)
  { return __builtin_tanhf(__x); }

  inline constexpr long double
  tanh(long double __x)
  { return __builtin_tanhl(__x); }


  template<typename _Tp>
    inline constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    double>::__type
    tanh(_Tp __x)
    { return __builtin_tanh(__x); }
# 536 "/usr/include/c++/10/cmath" 3
  constexpr int
  fpclassify(float __x)
  { return __builtin_fpclassify(0, 1, 4,
    3, 2, __x); }

  constexpr int
  fpclassify(double __x)
  { return __builtin_fpclassify(0, 1, 4,
    3, 2, __x); }

  constexpr int
  fpclassify(long double __x)
  { return __builtin_fpclassify(0, 1, 4,
    3, 2, __x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              int>::__type
    fpclassify(_Tp __x)
    { return __x != 0 ? 4 : 2; }



  constexpr bool
  isfinite(float __x)
  { return __builtin_isfinite(__x); }

  constexpr bool
  isfinite(double __x)
  { return __builtin_isfinite(__x); }

  constexpr bool
  isfinite(long double __x)
  { return __builtin_isfinite(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              bool>::__type
    isfinite(_Tp __x)
    { return true; }



  constexpr bool
  isinf(float __x)
  { return __builtin_isinf(__x); }





  constexpr bool
  isinf(double __x)
  { return __builtin_isinf(__x); }


  constexpr bool
  isinf(long double __x)
  { return __builtin_isinf(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              bool>::__type
    isinf(_Tp __x)
    { return false; }



  constexpr bool
  isnan(float __x)
  { return __builtin_isnan(__x); }





  constexpr bool
  isnan(double __x)
  { return __builtin_isnan(__x); }


  constexpr bool
  isnan(long double __x)
  { return __builtin_isnan(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              bool>::__type
    isnan(_Tp __x)
    { return false; }



  constexpr bool
  isnormal(float __x)
  { return __builtin_isnormal(__x); }

  constexpr bool
  isnormal(double __x)
  { return __builtin_isnormal(__x); }

  constexpr bool
  isnormal(long double __x)
  { return __builtin_isnormal(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              bool>::__type
    isnormal(_Tp __x)
    { return __x != 0 ? true : false; }




  constexpr bool
  signbit(float __x)
  { return __builtin_signbit(__x); }

  constexpr bool
  signbit(double __x)
  { return __builtin_signbit(__x); }

  constexpr bool
  signbit(long double __x)
  { return __builtin_signbit(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              bool>::__type
    signbit(_Tp __x)
    { return __x < 0 ? true : false; }



  constexpr bool
  isgreater(float __x, float __y)
  { return __builtin_isgreater(__x, __y); }

  constexpr bool
  isgreater(double __x, double __y)
  { return __builtin_isgreater(__x, __y); }

  constexpr bool
  isgreater(long double __x, long double __y)
  { return __builtin_isgreater(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename
    __gnu_cxx::__enable_if<(__is_arithmetic<_Tp>::__value
       && __is_arithmetic<_Up>::__value), bool>::__type
    isgreater(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return __builtin_isgreater(__type(__x), __type(__y));
    }



  constexpr bool
  isgreaterequal(float __x, float __y)
  { return __builtin_isgreaterequal(__x, __y); }

  constexpr bool
  isgreaterequal(double __x, double __y)
  { return __builtin_isgreaterequal(__x, __y); }

  constexpr bool
  isgreaterequal(long double __x, long double __y)
  { return __builtin_isgreaterequal(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename
    __gnu_cxx::__enable_if<(__is_arithmetic<_Tp>::__value
       && __is_arithmetic<_Up>::__value), bool>::__type
    isgreaterequal(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return __builtin_isgreaterequal(__type(__x), __type(__y));
    }



  constexpr bool
  isless(float __x, float __y)
  { return __builtin_isless(__x, __y); }

  constexpr bool
  isless(double __x, double __y)
  { return __builtin_isless(__x, __y); }

  constexpr bool
  isless(long double __x, long double __y)
  { return __builtin_isless(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename
    __gnu_cxx::__enable_if<(__is_arithmetic<_Tp>::__value
       && __is_arithmetic<_Up>::__value), bool>::__type
    isless(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return __builtin_isless(__type(__x), __type(__y));
    }



  constexpr bool
  islessequal(float __x, float __y)
  { return __builtin_islessequal(__x, __y); }

  constexpr bool
  islessequal(double __x, double __y)
  { return __builtin_islessequal(__x, __y); }

  constexpr bool
  islessequal(long double __x, long double __y)
  { return __builtin_islessequal(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename
    __gnu_cxx::__enable_if<(__is_arithmetic<_Tp>::__value
       && __is_arithmetic<_Up>::__value), bool>::__type
    islessequal(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return __builtin_islessequal(__type(__x), __type(__y));
    }



  constexpr bool
  islessgreater(float __x, float __y)
  { return __builtin_islessgreater(__x, __y); }

  constexpr bool
  islessgreater(double __x, double __y)
  { return __builtin_islessgreater(__x, __y); }

  constexpr bool
  islessgreater(long double __x, long double __y)
  { return __builtin_islessgreater(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename
    __gnu_cxx::__enable_if<(__is_arithmetic<_Tp>::__value
       && __is_arithmetic<_Up>::__value), bool>::__type
    islessgreater(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return __builtin_islessgreater(__type(__x), __type(__y));
    }



  constexpr bool
  isunordered(float __x, float __y)
  { return __builtin_isunordered(__x, __y); }

  constexpr bool
  isunordered(double __x, double __y)
  { return __builtin_isunordered(__x, __y); }

  constexpr bool
  isunordered(long double __x, long double __y)
  { return __builtin_isunordered(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename
    __gnu_cxx::__enable_if<(__is_arithmetic<_Tp>::__value
       && __is_arithmetic<_Up>::__value), bool>::__type
    isunordered(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return __builtin_isunordered(__type(__x), __type(__y));
    }
# 1065 "/usr/include/c++/10/cmath" 3
  using ::double_t;
  using ::float_t;


  using ::acosh;
  using ::acoshf;
  using ::acoshl;

  using ::asinh;
  using ::asinhf;
  using ::asinhl;

  using ::atanh;
  using ::atanhf;
  using ::atanhl;

  using ::cbrt;
  using ::cbrtf;
  using ::cbrtl;

  using ::copysign;
  using ::copysignf;
  using ::copysignl;

  using ::erf;
  using ::erff;
  using ::erfl;

  using ::erfc;
  using ::erfcf;
  using ::erfcl;

  using ::exp2;
  using ::exp2f;
  using ::exp2l;

  using ::expm1;
  using ::expm1f;
  using ::expm1l;

  using ::fdim;
  using ::fdimf;
  using ::fdiml;

  using ::fma;
  using ::fmaf;
  using ::fmal;

  using ::fmax;
  using ::fmaxf;
  using ::fmaxl;

  using ::fmin;
  using ::fminf;
  using ::fminl;

  using ::hypot;
  using ::hypotf;
  using ::hypotl;

  using ::ilogb;
  using ::ilogbf;
  using ::ilogbl;

  using ::lgamma;
  using ::lgammaf;
  using ::lgammal;


  using ::llrint;
  using ::llrintf;
  using ::llrintl;

  using ::llround;
  using ::llroundf;
  using ::llroundl;


  using ::log1p;
  using ::log1pf;
  using ::log1pl;

  using ::log2;
  using ::log2f;
  using ::log2l;

  using ::logb;
  using ::logbf;
  using ::logbl;

  using ::lrint;
  using ::lrintf;
  using ::lrintl;

  using ::lround;
  using ::lroundf;
  using ::lroundl;

  using ::nan;
  using ::nanf;
  using ::nanl;

  using ::nearbyint;
  using ::nearbyintf;
  using ::nearbyintl;

  using ::nextafter;
  using ::nextafterf;
  using ::nextafterl;

  using ::nexttoward;
  using ::nexttowardf;
  using ::nexttowardl;

  using ::remainder;
  using ::remainderf;
  using ::remainderl;

  using ::remquo;
  using ::remquof;
  using ::remquol;

  using ::rint;
  using ::rintf;
  using ::rintl;

  using ::round;
  using ::roundf;
  using ::roundl;

  using ::scalbln;
  using ::scalblnf;
  using ::scalblnl;

  using ::scalbn;
  using ::scalbnf;
  using ::scalbnl;

  using ::tgamma;
  using ::tgammaf;
  using ::tgammal;

  using ::trunc;
  using ::truncf;
  using ::truncl;



  constexpr float
  acosh(float __x)
  { return __builtin_acoshf(__x); }

  constexpr long double
  acosh(long double __x)
  { return __builtin_acoshl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    acosh(_Tp __x)
    { return __builtin_acosh(__x); }



  constexpr float
  asinh(float __x)
  { return __builtin_asinhf(__x); }

  constexpr long double
  asinh(long double __x)
  { return __builtin_asinhl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    asinh(_Tp __x)
    { return __builtin_asinh(__x); }



  constexpr float
  atanh(float __x)
  { return __builtin_atanhf(__x); }

  constexpr long double
  atanh(long double __x)
  { return __builtin_atanhl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    atanh(_Tp __x)
    { return __builtin_atanh(__x); }



  constexpr float
  cbrt(float __x)
  { return __builtin_cbrtf(__x); }

  constexpr long double
  cbrt(long double __x)
  { return __builtin_cbrtl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    cbrt(_Tp __x)
    { return __builtin_cbrt(__x); }



  constexpr float
  copysign(float __x, float __y)
  { return __builtin_copysignf(__x, __y); }

  constexpr long double
  copysign(long double __x, long double __y)
  { return __builtin_copysignl(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    copysign(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return copysign(__type(__x), __type(__y));
    }



  constexpr float
  erf(float __x)
  { return __builtin_erff(__x); }

  constexpr long double
  erf(long double __x)
  { return __builtin_erfl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    erf(_Tp __x)
    { return __builtin_erf(__x); }



  constexpr float
  erfc(float __x)
  { return __builtin_erfcf(__x); }

  constexpr long double
  erfc(long double __x)
  { return __builtin_erfcl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    erfc(_Tp __x)
    { return __builtin_erfc(__x); }



  constexpr float
  exp2(float __x)
  { return __builtin_exp2f(__x); }

  constexpr long double
  exp2(long double __x)
  { return __builtin_exp2l(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    exp2(_Tp __x)
    { return __builtin_exp2(__x); }



  constexpr float
  expm1(float __x)
  { return __builtin_expm1f(__x); }

  constexpr long double
  expm1(long double __x)
  { return __builtin_expm1l(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    expm1(_Tp __x)
    { return __builtin_expm1(__x); }



  constexpr float
  fdim(float __x, float __y)
  { return __builtin_fdimf(__x, __y); }

  constexpr long double
  fdim(long double __x, long double __y)
  { return __builtin_fdiml(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    fdim(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return fdim(__type(__x), __type(__y));
    }



  constexpr float
  fma(float __x, float __y, float __z)
  { return __builtin_fmaf(__x, __y, __z); }

  constexpr long double
  fma(long double __x, long double __y, long double __z)
  { return __builtin_fmal(__x, __y, __z); }



  template<typename _Tp, typename _Up, typename _Vp>
    constexpr typename __gnu_cxx::__promote_3<_Tp, _Up, _Vp>::__type
    fma(_Tp __x, _Up __y, _Vp __z)
    {
      typedef typename __gnu_cxx::__promote_3<_Tp, _Up, _Vp>::__type __type;
      return fma(__type(__x), __type(__y), __type(__z));
    }



  constexpr float
  fmax(float __x, float __y)
  { return __builtin_fmaxf(__x, __y); }

  constexpr long double
  fmax(long double __x, long double __y)
  { return __builtin_fmaxl(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    fmax(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return fmax(__type(__x), __type(__y));
    }



  constexpr float
  fmin(float __x, float __y)
  { return __builtin_fminf(__x, __y); }

  constexpr long double
  fmin(long double __x, long double __y)
  { return __builtin_fminl(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    fmin(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return fmin(__type(__x), __type(__y));
    }



  constexpr float
  hypot(float __x, float __y)
  { return __builtin_hypotf(__x, __y); }

  constexpr long double
  hypot(long double __x, long double __y)
  { return __builtin_hypotl(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    hypot(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return hypot(__type(__x), __type(__y));
    }



  constexpr int
  ilogb(float __x)
  { return __builtin_ilogbf(__x); }

  constexpr int
  ilogb(long double __x)
  { return __builtin_ilogbl(__x); }



  template<typename _Tp>
    constexpr
    typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                    int>::__type
    ilogb(_Tp __x)
    { return __builtin_ilogb(__x); }



  constexpr float
  lgamma(float __x)
  { return __builtin_lgammaf(__x); }

  constexpr long double
  lgamma(long double __x)
  { return __builtin_lgammal(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    lgamma(_Tp __x)
    { return __builtin_lgamma(__x); }



  constexpr long long
  llrint(float __x)
  { return __builtin_llrintf(__x); }

  constexpr long long
  llrint(long double __x)
  { return __builtin_llrintl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              long long>::__type
    llrint(_Tp __x)
    { return __builtin_llrint(__x); }



  constexpr long long
  llround(float __x)
  { return __builtin_llroundf(__x); }

  constexpr long long
  llround(long double __x)
  { return __builtin_llroundl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              long long>::__type
    llround(_Tp __x)
    { return __builtin_llround(__x); }



  constexpr float
  log1p(float __x)
  { return __builtin_log1pf(__x); }

  constexpr long double
  log1p(long double __x)
  { return __builtin_log1pl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    log1p(_Tp __x)
    { return __builtin_log1p(__x); }




  constexpr float
  log2(float __x)
  { return __builtin_log2f(__x); }

  constexpr long double
  log2(long double __x)
  { return __builtin_log2l(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    log2(_Tp __x)
    { return __builtin_log2(__x); }



  constexpr float
  logb(float __x)
  { return __builtin_logbf(__x); }

  constexpr long double
  logb(long double __x)
  { return __builtin_logbl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    logb(_Tp __x)
    { return __builtin_logb(__x); }



  constexpr long
  lrint(float __x)
  { return __builtin_lrintf(__x); }

  constexpr long
  lrint(long double __x)
  { return __builtin_lrintl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              long>::__type
    lrint(_Tp __x)
    { return __builtin_lrint(__x); }



  constexpr long
  lround(float __x)
  { return __builtin_lroundf(__x); }

  constexpr long
  lround(long double __x)
  { return __builtin_lroundl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              long>::__type
    lround(_Tp __x)
    { return __builtin_lround(__x); }



  constexpr float
  nearbyint(float __x)
  { return __builtin_nearbyintf(__x); }

  constexpr long double
  nearbyint(long double __x)
  { return __builtin_nearbyintl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    nearbyint(_Tp __x)
    { return __builtin_nearbyint(__x); }



  constexpr float
  nextafter(float __x, float __y)
  { return __builtin_nextafterf(__x, __y); }

  constexpr long double
  nextafter(long double __x, long double __y)
  { return __builtin_nextafterl(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    nextafter(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return nextafter(__type(__x), __type(__y));
    }



  constexpr float
  nexttoward(float __x, long double __y)
  { return __builtin_nexttowardf(__x, __y); }

  constexpr long double
  nexttoward(long double __x, long double __y)
  { return __builtin_nexttowardl(__x, __y); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    nexttoward(_Tp __x, long double __y)
    { return __builtin_nexttoward(__x, __y); }



  constexpr float
  remainder(float __x, float __y)
  { return __builtin_remainderf(__x, __y); }

  constexpr long double
  remainder(long double __x, long double __y)
  { return __builtin_remainderl(__x, __y); }



  template<typename _Tp, typename _Up>
    constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    remainder(_Tp __x, _Up __y)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return remainder(__type(__x), __type(__y));
    }



  inline float
  remquo(float __x, float __y, int* __pquo)
  { return __builtin_remquof(__x, __y, __pquo); }

  inline long double
  remquo(long double __x, long double __y, int* __pquo)
  { return __builtin_remquol(__x, __y, __pquo); }



  template<typename _Tp, typename _Up>
    inline typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
    remquo(_Tp __x, _Up __y, int* __pquo)
    {
      typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
      return remquo(__type(__x), __type(__y), __pquo);
    }



  constexpr float
  rint(float __x)
  { return __builtin_rintf(__x); }

  constexpr long double
  rint(long double __x)
  { return __builtin_rintl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    rint(_Tp __x)
    { return __builtin_rint(__x); }



  constexpr float
  round(float __x)
  { return __builtin_roundf(__x); }

  constexpr long double
  round(long double __x)
  { return __builtin_roundl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    round(_Tp __x)
    { return __builtin_round(__x); }



  constexpr float
  scalbln(float __x, long __ex)
  { return __builtin_scalblnf(__x, __ex); }

  constexpr long double
  scalbln(long double __x, long __ex)
  { return __builtin_scalblnl(__x, __ex); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    scalbln(_Tp __x, long __ex)
    { return __builtin_scalbln(__x, __ex); }



  constexpr float
  scalbn(float __x, int __ex)
  { return __builtin_scalbnf(__x, __ex); }

  constexpr long double
  scalbn(long double __x, int __ex)
  { return __builtin_scalbnl(__x, __ex); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    scalbn(_Tp __x, int __ex)
    { return __builtin_scalbn(__x, __ex); }



  constexpr float
  tgamma(float __x)
  { return __builtin_tgammaf(__x); }

  constexpr long double
  tgamma(long double __x)
  { return __builtin_tgammal(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    tgamma(_Tp __x)
    { return __builtin_tgamma(__x); }



  constexpr float
  trunc(float __x)
  { return __builtin_truncf(__x); }

  constexpr long double
  trunc(long double __x)
  { return __builtin_truncl(__x); }



  template<typename _Tp>
    constexpr typename __gnu_cxx::__enable_if<__is_integer<_Tp>::__value,
                                              double>::__type
    trunc(_Tp __x)
    { return __builtin_trunc(__x); }
# 1923 "/usr/include/c++/10/cmath" 3

}





}
# 37 "/usr/include/c++/10/math.h" 2 3

using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cos;
using std::sin;
using std::tan;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;


using std::fpclassify;
using std::isfinite;
using std::isinf;
using std::isnan;
using std::isnormal;
using std::signbit;
using std::isgreater;
using std::isgreaterequal;
using std::isless;
using std::islessequal;
using std::islessgreater;
using std::isunordered;



using std::acosh;
using std::asinh;
using std::atanh;
using std::cbrt;
using std::copysign;
using std::erf;
using std::erfc;
using std::exp2;
using std::expm1;
using std::fdim;
using std::fma;
using std::fmax;
using std::fmin;
using std::hypot;
using std::ilogb;
using std::lgamma;
using std::llrint;
using std::llround;
using std::log1p;
using std::log2;
using std::logb;
using std::lrint;
using std::lround;
using std::nearbyint;
using std::nextafter;
using std::nexttoward;
using std::remainder;
using std::remquo;
using std::rint;
using std::round;
using std::scalbln;
using std::scalbn;
using std::tgamma;
using std::trunc;
# 8955 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 2
# 1 "/usr/include/c++/10/stdlib.h" 1 3
# 8956 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 2


# 1 "/usr/include/c++/10/cmath" 1 3
# 39 "/usr/include/c++/10/cmath" 3
       
# 40 "/usr/include/c++/10/cmath" 3
# 8959 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 2
# 1 "/usr/include/c++/10/cstdlib" 1 3
# 39 "/usr/include/c++/10/cstdlib" 3
       
# 40 "/usr/include/c++/10/cstdlib" 3
# 8960 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 2
# 9029 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"

# 9029 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std {
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool signbit(float x);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool signbit(double x);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool signbit(long double x);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool isfinite(float x);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool isfinite(double x);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool isfinite(long double x);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool isnan(float x);




__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool isnan(double x);

__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool isnan(long double x);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool isinf(float x);




__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool isinf(double x);

__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr bool isinf(long double x);
}
# 9193 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std
{
  template<typename T> extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) T __pow_helper(T, int);
  template<typename T> extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) T __cmath_power(T, unsigned int);
}

using std::abs;
using std::fabs;
using std::ceil;
using std::floor;
using std::sqrt;

using std::pow;

using std::log;
using std::log10;
using std::fmod;
using std::modf;
using std::exp;
using std::frexp;
using std::ldexp;
using std::asin;
using std::sin;
using std::sinh;
using std::acos;
using std::cos;
using std::cosh;
using std::atan;
using std::atan2;
using std::tan;
using std::tanh;
# 9588 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std {
# 9597 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long long int abs(long long int);
# 9607 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long int abs(long int);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float abs(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double abs(double);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float fabs(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float ceil(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float floor(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float sqrt(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float pow(float, float);




template<typename _Tp, typename _Up>
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin))
typename __gnu_cxx::__promote_2<_Tp, _Up>::__type pow(_Tp, _Up);







extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float log(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float log10(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float fmod(float, float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float modf(float, float*);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float exp(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float frexp(float, int*);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float ldexp(float, int);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float asin(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float sin(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float sinh(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float acos(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float cos(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float cosh(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float atan(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float atan2(float, float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float tan(float);
extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float tanh(float);
# 9728 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}
# 9831 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std {
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float logb(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr int ilogb(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float scalbn(float a, int b);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float scalbln(float a, long int b);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float exp2(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float expm1(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float log2(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float log1p(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float acosh(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float asinh(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float atanh(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float hypot(float a, float b);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float cbrt(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float erf(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float erfc(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float lgamma(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float tgamma(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float copysign(float a, float b);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float nextafter(float a, float b);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float remainder(float a, float b);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float remquo(float a, float b, int *quo);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float round(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr long int lround(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr long long int llround(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float trunc(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float rint(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr long int lrint(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr long long int llrint(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float nearbyint(float a);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float fdim(float a, float b);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float fma(float a, float b, float c);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float fmax(float a, float b);
__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float fmin(float a, float b);
}
# 9970 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float exp10(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float rsqrt(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float rcbrt(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float sinpi(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float cospi(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void sincospi(float a, float *sptr, float *cptr);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) void sincos(float a, float *sptr, float *cptr);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float j0(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float j1(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float jn(int n, float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float y0(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float y1(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float yn(int n, float a);

static inline __attribute__((device)) __attribute__((cudart_builtin)) float cyl_bessel_i0(float a);

static inline __attribute__((device)) __attribute__((cudart_builtin)) float cyl_bessel_i1(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float erfinv(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float erfcinv(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float normcdfinv(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float normcdf(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float erfcx(float a);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double copysign(double a, float b);

static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double copysign(float a, double b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int min(unsigned int a, unsigned int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int min(int a, unsigned int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int min(unsigned int a, int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long int min(long int a, long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int min(unsigned long int a, unsigned long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int min(long int a, unsigned long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int min(unsigned long int a, long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long long int min(long long int a, long long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int min(unsigned long long int a, unsigned long long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int min(long long int a, unsigned long long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int min(unsigned long long int a, long long int b);
# 10111 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float min(float a, float b);
# 10122 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double min(double a, double b);
# 10132 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double min(float a, double b);
# 10142 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double min(double a, float b);
# 10153 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int max(unsigned int a, unsigned int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int max(int a, unsigned int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned int max(unsigned int a, int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long int max(long int a, long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int max(unsigned long int a, unsigned long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int max(long int a, unsigned long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long int max(unsigned long int a, long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) long long int max(long long int a, long long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int max(unsigned long long int a, unsigned long long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int max(long long int a, unsigned long long int b);







static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int max(unsigned long long int a, long long int b);
# 10244 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float max(float a, float b);
# 10255 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double max(double a, double b);
# 10265 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double max(float a, double b);
# 10275 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) double max(double a, float b);
# 10286 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern "C"{
inline __attribute__((device)) void *__nv_aligned_device_malloc(size_t size, size_t align)
{
  __attribute__((device)) void *__nv_aligned_device_malloc_impl(size_t, size_t);
  return __nv_aligned_device_malloc_impl(size, align);
}
}
# 10572 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.hpp" 1
# 77 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 1
# 78 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.hpp" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 79 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.hpp" 2
# 758 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
static inline __attribute__((host)) __attribute__((device)) float exp10(const float a)
{
  return exp10f(a);
}

static inline __attribute__((host)) __attribute__((device)) float rsqrt(const float a)
{
  return rsqrtf(a);
}

static inline __attribute__((host)) __attribute__((device)) float rcbrt(const float a)
{
  return rcbrtf(a);
}

static inline __attribute__((host)) __attribute__((device)) float sinpi(const float a)
{
  return sinpif(a);
}

static inline __attribute__((host)) __attribute__((device)) float cospi(const float a)
{
  return cospif(a);
}

static inline __attribute__((host)) __attribute__((device)) void sincospi(const float a, float *const sptr, float *const cptr)
{
  sincospif(a, sptr, cptr);
}

static inline __attribute__((host)) __attribute__((device)) void sincos(const float a, float *const sptr, float *const cptr)
{
  sincosf(a, sptr, cptr);
}

static inline __attribute__((host)) __attribute__((device)) float j0(const float a)
{
  return j0f(a);
}

static inline __attribute__((host)) __attribute__((device)) float j1(const float a)
{
  return j1f(a);
}

static inline __attribute__((host)) __attribute__((device)) float jn(const int n, const float a)
{
  return jnf(n, a);
}

static inline __attribute__((host)) __attribute__((device)) float y0(const float a)
{
  return y0f(a);
}

static inline __attribute__((host)) __attribute__((device)) float y1(const float a)
{
  return y1f(a);
}

static inline __attribute__((host)) __attribute__((device)) float yn(const int n, const float a)
{
  return ynf(n, a);
}

static inline __attribute__((device)) float cyl_bessel_i0(const float a)
{
  return cyl_bessel_i0f(a);
}

static inline __attribute__((device)) float cyl_bessel_i1(const float a)
{
  return cyl_bessel_i1f(a);
}

static inline __attribute__((host)) __attribute__((device)) float erfinv(const float a)
{
  return erfinvf(a);
}

static inline __attribute__((host)) __attribute__((device)) float erfcinv(const float a)
{
  return erfcinvf(a);
}

static inline __attribute__((host)) __attribute__((device)) float normcdfinv(const float a)
{
  return normcdfinvf(a);
}

static inline __attribute__((host)) __attribute__((device)) float normcdf(const float a)
{
  return normcdff(a);
}

static inline __attribute__((host)) __attribute__((device)) float erfcx(const float a)
{
  return erfcxf(a);
}

static inline __attribute__((host)) __attribute__((device)) double copysign(const double a, const float b)
{
  return copysign(a, static_cast<double>(b));
}

static inline __attribute__((host)) __attribute__((device)) double copysign(const float a, const double b)
{
  return copysign(static_cast<double>(a), b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned int min(const unsigned int a, const unsigned int b)
{
  return umin(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned int min(const int a, const unsigned int b)
{
  return umin(static_cast<unsigned int>(a), b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned int min(const unsigned int a, const int b)
{
  return umin(a, static_cast<unsigned int>(b));
}

static inline __attribute__((host)) __attribute__((device)) long int min(const long int a, const long int b)
{
  long int retval;





  if (sizeof(long int) == sizeof(int)) {



    retval = static_cast<long int>(min(static_cast<int>(a), static_cast<int>(b)));
  } else {
    retval = static_cast<long int>(llmin(static_cast<long long int>(a), static_cast<long long int>(b)));
  }
  return retval;
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int min(const unsigned long int a, const unsigned long int b)
{
  unsigned long int retval;



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    retval = static_cast<unsigned long int>(umin(static_cast<unsigned int>(a), static_cast<unsigned int>(b)));
  } else {
    retval = static_cast<unsigned long int>(ullmin(static_cast<unsigned long long int>(a), static_cast<unsigned long long int>(b)));
  }
  return retval;
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int min(const long int a, const unsigned long int b)
{
  unsigned long int retval;



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    retval = static_cast<unsigned long int>(umin(static_cast<unsigned int>(a), static_cast<unsigned int>(b)));
  } else {
    retval = static_cast<unsigned long int>(ullmin(static_cast<unsigned long long int>(a), static_cast<unsigned long long int>(b)));
  }
  return retval;
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int min(const unsigned long int a, const long int b)
{
  unsigned long int retval;



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    retval = static_cast<unsigned long int>(umin(static_cast<unsigned int>(a), static_cast<unsigned int>(b)));
  } else {
    retval = static_cast<unsigned long int>(ullmin(static_cast<unsigned long long int>(a), static_cast<unsigned long long int>(b)));
  }
  return retval;
}

static inline __attribute__((host)) __attribute__((device)) long long int min(const long long int a, const long long int b)
{
  return llmin(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int min(const unsigned long long int a, const unsigned long long int b)
{
  return ullmin(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int min(const long long int a, const unsigned long long int b)
{
  return ullmin(static_cast<unsigned long long int>(a), b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int min(const unsigned long long int a, const long long int b)
{
  return ullmin(a, static_cast<unsigned long long int>(b));
}

static inline __attribute__((host)) __attribute__((device)) float min(const float a, const float b)
{
  return fminf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) double min(const double a, const double b)
{
  return fmin(a, b);
}

static inline __attribute__((host)) __attribute__((device)) double min(const float a, const double b)
{
  return fmin(static_cast<double>(a), b);
}

static inline __attribute__((host)) __attribute__((device)) double min(const double a, const float b)
{
  return fmin(a, static_cast<double>(b));
}

static inline __attribute__((host)) __attribute__((device)) unsigned int max(const unsigned int a, const unsigned int b)
{
  return umax(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned int max(const int a, const unsigned int b)
{
  return umax(static_cast<unsigned int>(a), b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned int max(const unsigned int a, const int b)
{
  return umax(a, static_cast<unsigned int>(b));
}

static inline __attribute__((host)) __attribute__((device)) long int max(const long int a, const long int b)
{
  long int retval;




  if (sizeof(long int) == sizeof(int)) {



    retval = static_cast<long int>(max(static_cast<int>(a), static_cast<int>(b)));
  } else {
    retval = static_cast<long int>(llmax(static_cast<long long int>(a), static_cast<long long int>(b)));
  }
  return retval;
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int max(const unsigned long int a, const unsigned long int b)
{
  unsigned long int retval;



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    retval = static_cast<unsigned long int>(umax(static_cast<unsigned int>(a), static_cast<unsigned int>(b)));
  } else {
    retval = static_cast<unsigned long int>(ullmax(static_cast<unsigned long long int>(a), static_cast<unsigned long long int>(b)));
  }
  return retval;
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int max(const long int a, const unsigned long int b)
{
  unsigned long int retval;



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    retval = static_cast<unsigned long int>(umax(static_cast<unsigned int>(a), static_cast<unsigned int>(b)));
  } else {
    retval = static_cast<unsigned long int>(ullmax(static_cast<unsigned long long int>(a), static_cast<unsigned long long int>(b)));
  }
  return retval;
}

static inline __attribute__((host)) __attribute__((device)) unsigned long int max(const unsigned long int a, const long int b)
{
  unsigned long int retval;



  if (sizeof(unsigned long int) == sizeof(unsigned int)) {



    retval = static_cast<unsigned long int>(umax(static_cast<unsigned int>(a), static_cast<unsigned int>(b)));
  } else {
    retval = static_cast<unsigned long int>(ullmax(static_cast<unsigned long long int>(a), static_cast<unsigned long long int>(b)));
  }
  return retval;
}

static inline __attribute__((host)) __attribute__((device)) long long int max(const long long int a, const long long int b)
{
  return llmax(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int max(const unsigned long long int a, const unsigned long long int b)
{
  return ullmax(a, b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int max(const long long int a, const unsigned long long int b)
{
  return ullmax(static_cast<unsigned long long int>(a), b);
}

static inline __attribute__((host)) __attribute__((device)) unsigned long long int max(const unsigned long long int a, const long long int b)
{
  return ullmax(a, static_cast<unsigned long long int>(b));
}

static inline __attribute__((host)) __attribute__((device)) float max(const float a, const float b)
{
  return fmaxf(a, b);
}

static inline __attribute__((host)) __attribute__((device)) double max(const double a, const double b)
{
  return fmax(a, b);
}

static inline __attribute__((host)) __attribute__((device)) double max(const float a, const double b)
{
  return fmax(static_cast<double>(a), b);
}

static inline __attribute__((host)) __attribute__((device)) double max(const double a, const float b)
{
  return fmax(a, static_cast<double>(b));
}
# 10573 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/math_functions.h" 2
# 296 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/common_functions.h" 2
# 116 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_surface_types.h" 1
# 74 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_surface_types.h"
template<class T, int dim = 1>
struct __attribute__((device_builtin_surface_type)) surface : public surfaceReference
{

  __attribute__((host)) surface(void)
  {
    channelDesc = cudaCreateChannelDesc<T>();
  }

  __attribute__((host)) surface(struct cudaChannelFormatDesc desc)
  {
    channelDesc = desc;
  }

};

template<int dim>
struct __attribute__((device_builtin_surface_type)) surface<void, dim> : public surfaceReference
{

  __attribute__((host)) surface(void)
  {
    channelDesc = cudaCreateChannelDesc<void>();
  }

};
# 117 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_texture_types.h" 1
# 74 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_texture_types.h"
template<class T, int texType = 0x01, enum cudaTextureReadMode mode = cudaReadModeElementType>
struct __attribute__((device_builtin_texture_type)) texture : public textureReference
{

  __attribute__((host)) texture(int norm = 0,
                   enum cudaTextureFilterMode fMode = cudaFilterModePoint,
                   enum cudaTextureAddressMode aMode = cudaAddressModeClamp)
  {
    normalized = norm;
    filterMode = fMode;
    addressMode[0] = aMode;
    addressMode[1] = aMode;
    addressMode[2] = aMode;
    channelDesc = cudaCreateChannelDesc<T>();
    sRGB = 0;
  }

  __attribute__((host)) texture(int norm,
                   enum cudaTextureFilterMode fMode,
                   enum cudaTextureAddressMode aMode,
                   struct cudaChannelFormatDesc desc)
  {
    normalized = norm;
    filterMode = fMode;
    addressMode[0] = aMode;
    addressMode[1] = aMode;
    addressMode[2] = aMode;
    channelDesc = desc;
    sRGB = 0;
  }

};
# 118 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 1
# 79 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 1
# 80 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_types.h" 1
# 81 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 82 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2







extern "C"
{
# 100 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __mulhi(int x, int y);
# 110 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __umulhi(unsigned int x, unsigned int y);
# 120 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __mul64hi(long long int x, long long int y);
# 130 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y);
# 139 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __int_as_float(int x);
# 148 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __float_as_int(float x);
# 157 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __uint_as_float(unsigned int x);
# 166 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __float_as_uint(float x);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) void __syncthreads(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) void __prof_trigger(int);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) void __threadfence(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) void __threadfence_block(void);
__attribute__((device)) __attribute__((cudart_builtin))

__attribute__((__noreturn__))



__attribute__((device_builtin)) void __trap(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) void __brkpt();
# 201 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __saturatef(float x);
# 270 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __sad(int x, int y, unsigned int z);
# 338 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __usad(unsigned int x, unsigned int y, unsigned int z);
# 348 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __mul24(int x, int y);
# 358 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __umul24(unsigned int x, unsigned int y);
# 371 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float fdividef(float x, float y);
# 446 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fdividef(float x, float y);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) double fdivide(double x, double y);
# 459 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __sinf(float x) 
# 459 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 3 4
                                                                                                      throw ()
# 459 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
                                                                                                             ;
# 471 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __cosf(float x) 
# 471 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 3 4
                                                                                                      throw ()
# 471 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
                                                                                                             ;
# 485 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __tanf(float x) 
# 485 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 3 4
                                                                                                      throw ()
# 485 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
                                                                                                             ;
# 500 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) void __sincosf(float x, float *sptr, float *cptr) 
# 500 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 3 4
                                                                                                                                   throw ()
# 500 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
                                                                                                                                          ;
# 550 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __expf(float x) 
# 550 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 3 4
                                                                                                      throw ()
# 550 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
                                                                                                             ;
# 582 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __exp10f(float x) 
# 582 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 3 4
                                                                                                        throw ()
# 582 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
                                                                                                               ;
# 608 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __log2f(float x) 
# 608 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 3 4
                                                                                                       throw ()
# 608 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
                                                                                                              ;
# 636 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __log10f(float x) 
# 636 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 3 4
                                                                                                        throw ()
# 636 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
                                                                                                               ;
# 680 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __logf(float x) 
# 680 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 3 4
                                                                                                      throw ()
# 680 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
                                                                                                             ;
# 723 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) __attribute__((cudart_builtin)) float __powf(float x, float y) 
# 723 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 3 4
                                                                                                               throw ()
# 723 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
                                                                                                                      ;
# 732 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __float2int_rn(float x);
# 741 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __float2int_rz(float x);
# 750 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __float2int_ru(float);
# 759 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __float2int_rd(float x);
# 768 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __float2uint_rn(float x);
# 777 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __float2uint_rz(float x);
# 786 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __float2uint_ru(float x);
# 795 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __float2uint_rd(float x);
# 804 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __int2float_rn(int x);
# 813 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __int2float_rz(int x);
# 822 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __int2float_ru(int x);
# 831 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __int2float_rd(int x);
# 840 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __uint2float_rn(unsigned int x);
# 849 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __uint2float_rz(unsigned int x);
# 858 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __uint2float_ru(unsigned int x);
# 867 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __uint2float_rd(unsigned int x);
# 876 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __float2ll_rn(float x);
# 885 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __float2ll_rz(float x);
# 894 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __float2ll_ru(float x);
# 903 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __float2ll_rd(float x);
# 912 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __float2ull_rn(float x);
# 921 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __float2ull_rz(float x);
# 930 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __float2ull_ru(float x);
# 939 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __float2ull_rd(float x);
# 948 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ll2float_rn(long long int x);
# 957 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ll2float_rz(long long int x);
# 966 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ll2float_ru(long long int x);
# 975 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ll2float_rd(long long int x);
# 984 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ull2float_rn(unsigned long long int x);
# 993 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ull2float_rz(unsigned long long int x);
# 1002 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ull2float_ru(unsigned long long int x);
# 1011 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __ull2float_rd(unsigned long long int x);
# 1023 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fadd_rn(float x, float y);
# 1035 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fadd_rz(float x, float y);
# 1047 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fadd_ru(float x, float y);
# 1059 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fadd_rd(float x, float y);
# 1071 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsub_rn(float x, float y);
# 1083 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsub_rz(float x, float y);
# 1095 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsub_ru(float x, float y);
# 1107 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsub_rd(float x, float y);
# 1119 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmul_rn(float x, float y);
# 1131 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmul_rz(float x, float y);
# 1143 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmul_ru(float x, float y);
# 1155 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmul_rd(float x, float y);
# 1308 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmaf_rn(float x, float y, float z);
# 1461 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmaf_rz(float x, float y, float z);
# 1614 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmaf_ru(float x, float y, float z);
# 1767 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fmaf_rd(float x, float y, float z);
# 1800 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __frcp_rn(float x);
# 1833 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __frcp_rz(float x);
# 1866 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __frcp_ru(float x);
# 1899 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __frcp_rd(float x);
# 1930 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsqrt_rn(float x);
# 1961 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsqrt_rz(float x);
# 1992 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsqrt_ru(float x);
# 2023 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fsqrt_rd(float x);
# 2062 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __frsqrt_rn(float x);
# 2073 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fdiv_rn(float x, float y);
# 2084 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fdiv_rz(float x, float y);
# 2095 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fdiv_ru(float x, float y);
# 2106 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) float __fdiv_rd(float x, float y);
# 2115 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __clz(int x);
# 2126 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __ffs(int x);
# 2135 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __popc(unsigned int x);
# 2144 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __brev(unsigned int x);
# 2153 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __clzll(long long int x);
# 2164 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __ffsll(long long int x);
# 2175 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __popcll(unsigned long long int x);
# 2184 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __brevll(unsigned long long int x);
# 2208 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);
# 2220 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __hadd(int, int);
# 2233 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __rhadd(int, int);
# 2245 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __uhadd(unsigned int, unsigned int);
# 2258 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __urhadd(unsigned int, unsigned int);
# 2268 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) int __double2int_rz(double);
# 2277 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __double2uint_rz(double);
# 2286 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) long long int __double2ll_rz(double);
# 2295 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned long long int __double2ull_rz(double);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __pm0(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __pm1(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __pm2(void);
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __pm3(void);
# 2325 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabs2(unsigned int a);
# 2336 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsss2(unsigned int a);
# 2347 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vadd2(unsigned int a, unsigned int b);
# 2358 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vaddss2 (unsigned int a, unsigned int b);
# 2368 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vaddus2 (unsigned int a, unsigned int b);
# 2379 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vavgs2(unsigned int a, unsigned int b);
# 2390 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vavgu2(unsigned int a, unsigned int b);
# 2401 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vhaddu2(unsigned int a, unsigned int b);
# 2412 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpeq2(unsigned int a, unsigned int b);
# 2423 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpges2(unsigned int a, unsigned int b);
# 2434 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgeu2(unsigned int a, unsigned int b);
# 2445 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgts2(unsigned int a, unsigned int b);
# 2456 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgtu2(unsigned int a, unsigned int b);
# 2467 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmples2(unsigned int a, unsigned int b);
# 2479 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpleu2(unsigned int a, unsigned int b);
# 2490 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmplts2(unsigned int a, unsigned int b);
# 2501 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpltu2(unsigned int a, unsigned int b);
# 2512 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpne2(unsigned int a, unsigned int b);
# 2523 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsdiffu2(unsigned int a, unsigned int b);
# 2534 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmaxs2(unsigned int a, unsigned int b);
# 2545 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmaxu2(unsigned int a, unsigned int b);
# 2556 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmins2(unsigned int a, unsigned int b);
# 2567 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vminu2(unsigned int a, unsigned int b);
# 2578 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vseteq2(unsigned int a, unsigned int b);
# 2589 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetges2(unsigned int a, unsigned int b);
# 2600 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgeu2(unsigned int a, unsigned int b);
# 2611 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgts2(unsigned int a, unsigned int b);
# 2622 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgtu2(unsigned int a, unsigned int b);
# 2633 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetles2(unsigned int a, unsigned int b);
# 2644 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetleu2(unsigned int a, unsigned int b);
# 2655 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetlts2(unsigned int a, unsigned int b);
# 2666 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetltu2(unsigned int a, unsigned int b);
# 2677 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetne2(unsigned int a, unsigned int b);
# 2688 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsadu2(unsigned int a, unsigned int b);
# 2699 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsub2(unsigned int a, unsigned int b);
# 2710 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsubss2 (unsigned int a, unsigned int b);
# 2721 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsubus2 (unsigned int a, unsigned int b);
# 2731 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vneg2(unsigned int a);
# 2741 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vnegss2(unsigned int a);
# 2752 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsdiffs2(unsigned int a, unsigned int b);
# 2763 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsads2(unsigned int a, unsigned int b);
# 2773 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabs4(unsigned int a);
# 2784 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsss4(unsigned int a);
# 2795 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vadd4(unsigned int a, unsigned int b);
# 2806 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vaddss4 (unsigned int a, unsigned int b);
# 2816 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vaddus4 (unsigned int a, unsigned int b);
# 2827 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vavgs4(unsigned int a, unsigned int b);
# 2838 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vavgu4(unsigned int a, unsigned int b);
# 2849 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vhaddu4(unsigned int a, unsigned int b);
# 2860 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpeq4(unsigned int a, unsigned int b);
# 2871 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpges4(unsigned int a, unsigned int b);
# 2882 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgeu4(unsigned int a, unsigned int b);
# 2893 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgts4(unsigned int a, unsigned int b);
# 2904 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpgtu4(unsigned int a, unsigned int b);
# 2915 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmples4(unsigned int a, unsigned int b);
# 2926 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpleu4(unsigned int a, unsigned int b);
# 2937 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmplts4(unsigned int a, unsigned int b);
# 2948 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpltu4(unsigned int a, unsigned int b);
# 2959 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vcmpne4(unsigned int a, unsigned int b);
# 2970 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsdiffu4(unsigned int a, unsigned int b);
# 2981 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmaxs4(unsigned int a, unsigned int b);
# 2992 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmaxu4(unsigned int a, unsigned int b);
# 3003 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vmins4(unsigned int a, unsigned int b);
# 3014 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vminu4(unsigned int a, unsigned int b);
# 3025 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vseteq4(unsigned int a, unsigned int b);
# 3036 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetles4(unsigned int a, unsigned int b);
# 3047 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetleu4(unsigned int a, unsigned int b);
# 3058 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetlts4(unsigned int a, unsigned int b);
# 3069 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetltu4(unsigned int a, unsigned int b);
# 3080 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetges4(unsigned int a, unsigned int b);
# 3091 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgeu4(unsigned int a, unsigned int b);
# 3102 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgts4(unsigned int a, unsigned int b);
# 3113 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetgtu4(unsigned int a, unsigned int b);
# 3124 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsetne4(unsigned int a, unsigned int b);
# 3135 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsadu4(unsigned int a, unsigned int b);
# 3146 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsub4(unsigned int a, unsigned int b);
# 3157 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsubss4(unsigned int a, unsigned int b);
# 3168 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsubus4(unsigned int a, unsigned int b);
# 3178 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vneg4(unsigned int a);
# 3188 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vnegss4(unsigned int a);
# 3199 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vabsdiffs4(unsigned int a, unsigned int b);
# 3210 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
__attribute__((device)) __attribute__((cudart_builtin)) __attribute__((device_builtin)) unsigned int __vsads4(unsigned int a, unsigned int b);






}







static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) int mulhi(int a, int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int mulhi(unsigned int a, unsigned int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int mulhi(int a, unsigned int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int mulhi(unsigned int a, int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) long long int mul64hi(long long int a, long long int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int mul64hi(unsigned long long int a, unsigned long long int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int mul64hi(long long int a, unsigned long long int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned long long int mul64hi(unsigned long long int a, long long int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) int float_as_int(float a);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) float int_as_float(int a);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int float_as_uint(float a);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) float uint_as_float(unsigned int a);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) float saturate(float a);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) int mul24(int a, int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int umul24(unsigned int a, unsigned int b);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) int float2int(float a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) unsigned int float2uint(float a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) float int2float(int a, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) __attribute__((cudart_builtin)) float uint2float(unsigned int a, enum cudaRoundMode mode = cudaRoundNearest);
# 3275 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp" 1
# 79 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 1
# 80 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp" 2

# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 82 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp" 2
# 90 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
static __inline__ __attribute__((device)) int mulhi(const int a, const int b)
{
  return __mulhi(a, b);
}

static __inline__ __attribute__((device)) unsigned int mulhi(const unsigned int a, const unsigned int b)
{
  return __umulhi(a, b);
}

static __inline__ __attribute__((device)) unsigned int mulhi(const int a, const unsigned int b)
{
  return __umulhi(static_cast<unsigned int>(a), b);
}

static __inline__ __attribute__((device)) unsigned int mulhi(const unsigned int a, const int b)
{
  return __umulhi(a, static_cast<unsigned int>(b));
}

static __inline__ __attribute__((device)) long long int mul64hi(const long long int a, const long long int b)
{
  return __mul64hi(a, b);
}

static __inline__ __attribute__((device)) unsigned long long int mul64hi(const unsigned long long int a, const unsigned long long int b)
{
  return __umul64hi(a, b);
}

static __inline__ __attribute__((device)) unsigned long long int mul64hi(const long long int a, const unsigned long long int b)
{
  return __umul64hi(static_cast<unsigned long long int>(a), b);
}

static __inline__ __attribute__((device)) unsigned long long int mul64hi(const unsigned long long int a, const long long int b)
{
  return __umul64hi(a, static_cast<unsigned long long int>(b));
}

static __inline__ __attribute__((device)) int float_as_int(const float a)
{
  return __float_as_int(a);
}

static __inline__ __attribute__((device)) float int_as_float(const int a)
{
  return __int_as_float(a);
}

static __inline__ __attribute__((device)) unsigned int float_as_uint(const float a)
{
  return __float_as_uint(a);
}

static __inline__ __attribute__((device)) float uint_as_float(const unsigned int a)
{
  return __uint_as_float(a);
}
static __inline__ __attribute__((device)) float saturate(const float a)
{
  return __saturatef(a);
}

static __inline__ __attribute__((device)) int mul24(const int a, const int b)
{
  return __mul24(a, b);
}

static __inline__ __attribute__((device)) unsigned int umul24(const unsigned int a, const unsigned int b)
{
  return __umul24(a, b);
}

static __inline__ __attribute__((device)) int float2int(const float a, const enum cudaRoundMode mode)
{
  return (mode == cudaRoundNearest) ? __float2int_rn(a) :
         (mode == cudaRoundPosInf ) ? __float2int_ru(a) :
         (mode == cudaRoundMinInf ) ? __float2int_rd(a) :
                                      __float2int_rz(a);
}

static __inline__ __attribute__((device)) unsigned int float2uint(const float a, const enum cudaRoundMode mode)
{
  return (mode == cudaRoundNearest) ? __float2uint_rn(a) :
         (mode == cudaRoundPosInf ) ? __float2uint_ru(a) :
         (mode == cudaRoundMinInf ) ? __float2uint_rd(a) :
                                      __float2uint_rz(a);
}

static __inline__ __attribute__((device)) float int2float(const int a, const enum cudaRoundMode mode)
{
  return (mode == cudaRoundZero ) ? __int2float_rz(a) :
         (mode == cudaRoundPosInf) ? __int2float_ru(a) :
         (mode == cudaRoundMinInf) ? __int2float_rd(a) :
                                     __int2float_rn(a);
}

static __inline__ __attribute__((device)) float uint2float(const unsigned int a, const enum cudaRoundMode mode)
{
  return (mode == cudaRoundZero ) ? __uint2float_rz(a) :
         (mode == cudaRoundPosInf) ? __uint2float_ru(a) :
         (mode == cudaRoundMinInf) ? __uint2float_rd(a) :
                                     __uint2float_rn(a);
}
# 3276 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2


# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h" 1
# 76 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
extern "C"
{
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicAdd(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicAdd(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicExch(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicExch(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) float __fAtomicExch(float *address, float val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicMin(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicMin(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicMax(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicMax(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicInc(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicDec(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicAnd(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicAnd(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicOr(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicOr(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicXor(int *address, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicXor(unsigned int *address, unsigned int val);
extern __attribute__((device)) __attribute__((device_builtin)) int __iAtomicCAS(int *address, int compare, int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __uAtomicCAS(unsigned int *address, unsigned int compare, unsigned int val);
}
# 106 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
static __inline__ __attribute__((device)) int atomicAdd(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicAdd(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicSub(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicSub(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicExch(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicExch(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) float atomicExch(float *address, float val) ;

static __inline__ __attribute__((device)) int atomicMin(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicMin(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicMax(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicMax(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) unsigned int atomicInc(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) unsigned int atomicDec(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicAnd(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicAnd(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicOr(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicOr(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicXor(int *address, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicXor(unsigned int *address, unsigned int val) ;

static __inline__ __attribute__((device)) int atomicCAS(int *address, int compare, int val) ;

static __inline__ __attribute__((device)) unsigned int atomicCAS(unsigned int *address, unsigned int compare, unsigned int val) ;
# 171 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
extern "C"
{

extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __ullAtomicAdd(unsigned long long int *address, unsigned long long int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __ullAtomicExch(unsigned long long int *address, unsigned long long int val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __ullAtomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val);

extern __attribute__((device)) __attribute__((device_builtin)) __attribute__((deprecated("__any""() is deprecated in favor of ""__any""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) int __any(int cond);
extern __attribute__((device)) __attribute__((device_builtin)) __attribute__((deprecated("__all""() is deprecated in favor of ""__all""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) int __all(int cond);
}
# 189 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
static __inline__ __attribute__((device)) unsigned long long int atomicAdd(unsigned long long int *address, unsigned long long int val) ;

static __inline__ __attribute__((device)) unsigned long long int atomicExch(unsigned long long int *address, unsigned long long int val) ;

static __inline__ __attribute__((device)) unsigned long long int atomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val) ;

static __inline__ __attribute__((device)) __attribute__((deprecated("__any""() is deprecated in favor of ""__any""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) bool any(bool cond) ;

static __inline__ __attribute__((device)) __attribute__((deprecated("__all""() is deprecated in favor of ""__all""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) bool all(bool cond) ;
# 208 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_atomic_functions.hpp" 1
# 75 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_atomic_functions.hpp"
static __inline__ __attribute__((device)) int atomicAdd(int *address, int val)
{
  return __iAtomicAdd(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicAdd(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd(address, val);
}

static __inline__ __attribute__((device)) int atomicSub(int *address, int val)
{
  return __iAtomicAdd(address, (unsigned int)-(int)val);
}

static __inline__ __attribute__((device)) unsigned int atomicSub(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd(address, (unsigned int)-(int)val);
}

static __inline__ __attribute__((device)) int atomicExch(int *address, int val)
{
  return __iAtomicExch(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicExch(unsigned int *address, unsigned int val)
{
  return __uAtomicExch(address, val);
}

static __inline__ __attribute__((device)) float atomicExch(float *address, float val)
{
  return __fAtomicExch(address, val);
}

static __inline__ __attribute__((device)) int atomicMin(int *address, int val)
{
  return __iAtomicMin(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicMin(unsigned int *address, unsigned int val)
{
  return __uAtomicMin(address, val);
}

static __inline__ __attribute__((device)) int atomicMax(int *address, int val)
{
  return __iAtomicMax(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicMax(unsigned int *address, unsigned int val)
{
  return __uAtomicMax(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicInc(unsigned int *address, unsigned int val)
{
  return __uAtomicInc(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicDec(unsigned int *address, unsigned int val)
{
  return __uAtomicDec(address, val);
}

static __inline__ __attribute__((device)) int atomicAnd(int *address, int val)
{
  return __iAtomicAnd(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicAnd(unsigned int *address, unsigned int val)
{
  return __uAtomicAnd(address, val);
}

static __inline__ __attribute__((device)) int atomicOr(int *address, int val)
{
  return __iAtomicOr(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicOr(unsigned int *address, unsigned int val)
{
  return __uAtomicOr(address, val);
}

static __inline__ __attribute__((device)) int atomicXor(int *address, int val)
{
  return __iAtomicXor(address, val);
}

static __inline__ __attribute__((device)) unsigned int atomicXor(unsigned int *address, unsigned int val)
{
  return __uAtomicXor(address, val);
}

static __inline__ __attribute__((device)) int atomicCAS(int *address, int compare, int val)
{
  return __iAtomicCAS(address, compare, val);
}

static __inline__ __attribute__((device)) unsigned int atomicCAS(unsigned int *address, unsigned int compare, unsigned int val)
{
  return __uAtomicCAS(address, compare, val);
}
# 194 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_atomic_functions.hpp"
static __inline__ __attribute__((device)) unsigned long long int atomicAdd(unsigned long long int *address, unsigned long long int val)
{
  return __ullAtomicAdd(address, val);
}

static __inline__ __attribute__((device)) unsigned long long int atomicExch(unsigned long long int *address, unsigned long long int val)
{
  return __ullAtomicExch(address, val);
}

static __inline__ __attribute__((device)) unsigned long long int atomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val)
{
  return __ullAtomicCAS(address, compare, val);
}

static __inline__ __attribute__((device)) bool any(bool cond)
{
  return (bool)__any((int)cond);
}

static __inline__ __attribute__((device)) bool all(bool cond)
{
  return (bool)__all((int)cond);
}
# 209 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_atomic_functions.h" 2
# 3279 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h" 1
# 83 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 1
# 84 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h" 2

# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 86 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h" 2

extern "C"
{
# 97 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double_as_longlong(double x);
# 106 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __longlong_as_double(long long int x);
# 263 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rn(double x, double y, double z);
# 420 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rz(double x, double y, double z);
# 577 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_ru(double x, double y, double z);
# 734 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rd(double x, double y, double z);
# 746 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rn(double x, double y);
# 758 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rz(double x, double y);
# 770 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_ru(double x, double y);
# 782 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rd(double x, double y);
# 794 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rn(double x, double y);
# 806 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rz(double x, double y);
# 818 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_ru(double x, double y);
# 830 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rd(double x, double y);
# 842 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rn(double x, double y);
# 854 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rz(double x, double y);
# 866 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_ru(double x, double y);
# 878 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rd(double x, double y);
# 887 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rn(double x);
# 896 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rz(double x);
# 905 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_ru(double x);
# 914 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rd(double x);
# 923 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_rn(double x);
# 932 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_ru(double x);
# 941 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_rd(double x);
# 950 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_rn(double x);
# 959 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_ru(double x);
# 968 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_rd(double x);
# 977 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_rn(double x);
# 986 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_ru(double x);
# 995 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_rd(double x);
# 1004 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_rn(double x);
# 1013 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_ru(double x);
# 1022 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_rd(double x);







extern __attribute__((device)) __attribute__((device_builtin)) double __int2double_rn(int x);







extern __attribute__((device)) __attribute__((device_builtin)) double __uint2double_rn(unsigned int x);
# 1047 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rn(long long int x);
# 1056 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rz(long long int x);
# 1065 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_ru(long long int x);
# 1074 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rd(long long int x);
# 1083 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rn(unsigned long long int x);
# 1092 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rz(unsigned long long int x);
# 1101 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_ru(unsigned long long int x);
# 1110 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rd(unsigned long long int x);
# 1119 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2hiint(double x);
# 1128 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2loint(double x);
# 1138 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __hiloint2double(int hi, int lo);
}







static __inline__ __attribute__((device)) double fma(double a, double b, double c, enum cudaRoundMode mode);

static __inline__ __attribute__((device)) double dmul(double a, double b, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double dadd(double a, double b, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double dsub(double a, double b, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) int double2int(double a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) unsigned int double2uint(double a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) long long int double2ll(double a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) unsigned long long int double2ull(double a, enum cudaRoundMode mode = cudaRoundZero);

static __inline__ __attribute__((device)) double ll2double(long long int a, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double ull2double(unsigned long long int a, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double int2double(int a, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double uint2double(unsigned int a, enum cudaRoundMode mode = cudaRoundNearest);

static __inline__ __attribute__((device)) double float2double(float a, enum cudaRoundMode mode = cudaRoundNearest);






# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp" 1
# 83 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/builtin_types.h" 1
# 84 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp" 2

# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/host_defines.h" 1
# 86 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp" 2







static __inline__ __attribute__((device)) double fma(double a, double b, double c, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __fma_rz(a, b, c) :
         mode == cudaRoundPosInf ? __fma_ru(a, b, c) :
         mode == cudaRoundMinInf ? __fma_rd(a, b, c) :
                                   __fma_rn(a, b, c);
}

static __inline__ __attribute__((device)) double dmul(double a, double b, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __dmul_rz(a, b) :
         mode == cudaRoundPosInf ? __dmul_ru(a, b) :
         mode == cudaRoundMinInf ? __dmul_rd(a, b) :
                                   __dmul_rn(a, b);
}

static __inline__ __attribute__((device)) double dadd(double a, double b, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __dadd_rz(a, b) :
         mode == cudaRoundPosInf ? __dadd_ru(a, b) :
         mode == cudaRoundMinInf ? __dadd_rd(a, b) :
                                   __dadd_rn(a, b);
}

static __inline__ __attribute__((device)) double dsub(double a, double b, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __dsub_rz(a, b) :
         mode == cudaRoundPosInf ? __dsub_ru(a, b) :
         mode == cudaRoundMinInf ? __dsub_rd(a, b) :
                                   __dsub_rn(a, b);
}

static __inline__ __attribute__((device)) int double2int(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2int_rn(a) :
         mode == cudaRoundPosInf ? __double2int_ru(a) :
         mode == cudaRoundMinInf ? __double2int_rd(a) :
                                    __double2int_rz(a);
}

static __inline__ __attribute__((device)) unsigned int double2uint(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2uint_rn(a) :
         mode == cudaRoundPosInf ? __double2uint_ru(a) :
         mode == cudaRoundMinInf ? __double2uint_rd(a) :
                                    __double2uint_rz(a);
}

static __inline__ __attribute__((device)) long long int double2ll(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2ll_rn(a) :
         mode == cudaRoundPosInf ? __double2ll_ru(a) :
         mode == cudaRoundMinInf ? __double2ll_rd(a) :
                                    __double2ll_rz(a);
}

static __inline__ __attribute__((device)) unsigned long long int double2ull(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2ull_rn(a) :
         mode == cudaRoundPosInf ? __double2ull_ru(a) :
         mode == cudaRoundMinInf ? __double2ull_rd(a) :
                                    __double2ull_rz(a);
}

static __inline__ __attribute__((device)) double ll2double(long long int a, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __ll2double_rz(a) :
         mode == cudaRoundPosInf ? __ll2double_ru(a) :
         mode == cudaRoundMinInf ? __ll2double_rd(a) :
                                   __ll2double_rn(a);
}

static __inline__ __attribute__((device)) double ull2double(unsigned long long int a, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero ? __ull2double_rz(a) :
         mode == cudaRoundPosInf ? __ull2double_ru(a) :
         mode == cudaRoundMinInf ? __ull2double_rd(a) :
                                   __ull2double_rn(a);
}

static __inline__ __attribute__((device)) double int2double(int a, enum cudaRoundMode mode)
{
  return (double)a;
}

static __inline__ __attribute__((device)) double uint2double(unsigned int a, enum cudaRoundMode mode)
{
  return (double)a;
}

static __inline__ __attribute__((device)) double float2double(float a, enum cudaRoundMode mode)
{
  return (double)a;
}
# 1179 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_double_functions.h" 2
# 3280 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.h" 1
# 77 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.h"
extern "C"
{
extern __attribute__((device)) __attribute__((device_builtin)) float __fAtomicAdd(float *address, float val);
}
# 89 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.h"
static __inline__ __attribute__((device)) float atomicAdd(float *address, float val) ;







# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.hpp" 1
# 75 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.hpp"
static __inline__ __attribute__((device)) float atomicAdd(float *address, float val)
{
  return __fAtomicAdd(address, val);
}
# 98 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.h" 2
# 3281 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h" 1
# 79 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
extern "C"
{
extern __attribute__((device)) __attribute__((device_builtin)) long long __illAtomicMin(long long *address, long long val);
extern __attribute__((device)) __attribute__((device_builtin)) long long __illAtomicMax(long long *address, long long val);
extern __attribute__((device)) __attribute__((device_builtin)) long long __llAtomicAnd(long long *address, long long val);
extern __attribute__((device)) __attribute__((device_builtin)) long long __llAtomicOr(long long *address, long long val);
extern __attribute__((device)) __attribute__((device_builtin)) long long __llAtomicXor(long long *address, long long val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long __ullAtomicMin(unsigned long long *address, unsigned long long val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long __ullAtomicMax(unsigned long long *address, unsigned long long val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long __ullAtomicAnd(unsigned long long *address, unsigned long long val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long __ullAtomicOr (unsigned long long *address, unsigned long long val);
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long __ullAtomicXor(unsigned long long *address, unsigned long long val);
}
# 100 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
static __inline__ __attribute__((device)) long long atomicMin(long long *address, long long val) ;

static __inline__ __attribute__((device)) long long atomicMax(long long *address, long long val) ;

static __inline__ __attribute__((device)) long long atomicAnd(long long *address, long long val) ;

static __inline__ __attribute__((device)) long long atomicOr(long long *address, long long val) ;

static __inline__ __attribute__((device)) long long atomicXor(long long *address, long long val) ;

static __inline__ __attribute__((device)) unsigned long long atomicMin(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device)) unsigned long long atomicMax(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device)) unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device)) unsigned long long atomicOr(unsigned long long *address, unsigned long long val) ;

static __inline__ __attribute__((device)) unsigned long long atomicXor(unsigned long long *address, unsigned long long val) ;
# 128 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.hpp" 1
# 77 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.hpp"
static __inline__ __attribute__((device)) long long atomicMin(long long *address, long long val)
{
    return __illAtomicMin(address, val);
}

static __inline__ __attribute__((device)) long long atomicMax(long long *address, long long val)
{
    return __illAtomicMax(address, val);
}

static __inline__ __attribute__((device)) long long atomicAnd(long long *address, long long val)
{
    return __llAtomicAnd(address, val);
}

static __inline__ __attribute__((device)) long long atomicOr(long long *address, long long val)
{
    return __llAtomicOr(address, val);
}

static __inline__ __attribute__((device)) long long atomicXor(long long *address, long long val)
{
    return __llAtomicXor(address, val);
}

static __inline__ __attribute__((device)) unsigned long long atomicMin(unsigned long long *address, unsigned long long val)
{
    return __ullAtomicMin(address, val);
}

static __inline__ __attribute__((device)) unsigned long long atomicMax(unsigned long long *address, unsigned long long val)
{
    return __ullAtomicMax(address, val);
}

static __inline__ __attribute__((device)) unsigned long long atomicAnd(unsigned long long *address, unsigned long long val)
{
    return __ullAtomicAnd(address, val);
}

static __inline__ __attribute__((device)) unsigned long long atomicOr(unsigned long long *address, unsigned long long val)
{
    return __ullAtomicOr(address, val);
}

static __inline__ __attribute__((device)) unsigned long long atomicXor(unsigned long long *address, unsigned long long val)
{
    return __ullAtomicXor(address, val);
}
# 129 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h" 2
# 3282 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_35_atomic_functions.h" 1
# 56 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_35_atomic_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h" 1
# 57 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_35_atomic_functions.h" 2
# 3283 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h" 1
# 535 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.hpp" 1
# 536 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h" 2
# 3284 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h" 1
# 90 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern "C"
{
extern __attribute__((device)) __attribute__((device_builtin)) void __threadfence_system(void);
# 104 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ddiv_rn(double x, double y);
# 116 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ddiv_rz(double x, double y);
# 128 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ddiv_ru(double x, double y);
# 140 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ddiv_rd(double x, double y);
# 174 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __drcp_rn(double x);
# 208 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __drcp_rz(double x);
# 242 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __drcp_ru(double x);
# 276 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __drcp_rd(double x);
# 308 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsqrt_rn(double x);
# 340 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsqrt_rz(double x);
# 372 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsqrt_ru(double x);
# 404 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsqrt_rd(double x);
extern __attribute__((device)) __attribute__((device_builtin)) __attribute__((deprecated("__ballot""() is deprecated in favor of ""__ballot""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned int __ballot(int);
extern __attribute__((device)) __attribute__((device_builtin)) int __syncthreads_count(int);
extern __attribute__((device)) __attribute__((device_builtin)) int __syncthreads_and(int);
extern __attribute__((device)) __attribute__((device_builtin)) int __syncthreads_or(int);
extern __attribute__((device)) __attribute__((device_builtin)) long long int clock64(void);
# 419 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __fmaf_ieee_rn(float x, float y, float z);
# 428 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __fmaf_ieee_rd(float x, float y, float z);
# 437 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __fmaf_ieee_ru(float x, float y, float z);
# 446 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __fmaf_ieee_rz(float x, float y, float z);
# 459 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double_as_longlong(double x);
# 468 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __longlong_as_double(long long int x);
# 625 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rn(double x, double y, double z);
# 782 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rz(double x, double y, double z);
# 939 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_ru(double x, double y, double z);
# 1096 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __fma_rd(double x, double y, double z);
# 1108 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rn(double x, double y);
# 1120 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rz(double x, double y);
# 1132 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_ru(double x, double y);
# 1144 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dadd_rd(double x, double y);
# 1156 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rn(double x, double y);
# 1168 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rz(double x, double y);
# 1180 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_ru(double x, double y);
# 1192 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dsub_rd(double x, double y);
# 1204 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rn(double x, double y);
# 1216 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rz(double x, double y);
# 1228 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_ru(double x, double y);
# 1240 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __dmul_rd(double x, double y);
# 1249 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rn(double x);
# 1258 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rz(double x);
# 1267 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_ru(double x);
# 1276 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) float __double2float_rd(double x);
# 1285 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_rn(double x);
# 1294 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_ru(double x);
# 1303 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2int_rd(double x);
# 1312 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_rn(double x);
# 1321 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_ru(double x);
# 1330 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __double2uint_rd(double x);
# 1339 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_rn(double x);
# 1348 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_ru(double x);
# 1357 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) long long int __double2ll_rd(double x);
# 1366 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_rn(double x);
# 1375 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_ru(double x);
# 1384 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) unsigned long long int __double2ull_rd(double x);







extern __attribute__((device)) __attribute__((device_builtin)) double __int2double_rn(int x);







extern __attribute__((device)) __attribute__((device_builtin)) double __uint2double_rn(unsigned int x);
# 1409 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rn(long long int x);
# 1418 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rz(long long int x);
# 1427 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_ru(long long int x);
# 1436 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ll2double_rd(long long int x);
# 1445 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rn(unsigned long long int x);
# 1454 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rz(unsigned long long int x);
# 1463 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_ru(unsigned long long int x);
# 1472 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __ull2double_rd(unsigned long long int x);
# 1481 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2hiint(double x);
# 1490 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) int __double2loint(double x);
# 1500 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern __attribute__((device)) __attribute__((device_builtin)) double __hiloint2double(int hi, int lo);


}






static __inline__ __attribute__((device)) __attribute__((deprecated("__ballot""() is deprecated in favor of ""__ballot""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned int ballot(bool pred) ;

static __inline__ __attribute__((device)) int syncthreads_count(bool pred) ;

static __inline__ __attribute__((device)) bool syncthreads_and(bool pred) ;

static __inline__ __attribute__((device)) bool syncthreads_or(bool pred) ;




static __inline__ __attribute__((device)) unsigned int __isGlobal(const void *ptr) ;
static __inline__ __attribute__((device)) unsigned int __isShared(const void *ptr) ;
static __inline__ __attribute__((device)) unsigned int __isConstant(const void *ptr) ;
static __inline__ __attribute__((device)) unsigned int __isLocal(const void *ptr) ;

static __inline__ __attribute__((device)) size_t __cvta_generic_to_global(const void *ptr) ;
static __inline__ __attribute__((device)) size_t __cvta_generic_to_shared(const void *ptr) ;
static __inline__ __attribute__((device)) size_t __cvta_generic_to_constant(const void *ptr) ;
static __inline__ __attribute__((device)) size_t __cvta_generic_to_local(const void *ptr) ;

static __inline__ __attribute__((device)) void * __cvta_global_to_generic(size_t rawbits) ;
static __inline__ __attribute__((device)) void * __cvta_shared_to_generic(size_t rawbits) ;
static __inline__ __attribute__((device)) void * __cvta_constant_to_generic(size_t rawbits) ;
static __inline__ __attribute__((device)) void * __cvta_local_to_generic(size_t rawbits) ;







# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.hpp" 1
# 75 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.hpp"
static __inline__ __attribute__((device)) unsigned int ballot(bool pred)
{
  return __ballot((int)pred);
}

static __inline__ __attribute__((device)) int syncthreads_count(bool pred)
{
  return __syncthreads_count((int)pred);
}

static __inline__ __attribute__((device)) bool syncthreads_and(bool pred)
{
  return (bool)__syncthreads_and((int)pred);
}

static __inline__ __attribute__((device)) bool syncthreads_or(bool pred)
{
  return (bool)__syncthreads_or((int)pred);
}


extern "C" {
  __attribute__((device)) unsigned __nv_isGlobal_impl(const void *);
  __attribute__((device)) unsigned __nv_isShared_impl(const void *);
  __attribute__((device)) unsigned __nv_isConstant_impl(const void *);
  __attribute__((device)) unsigned __nv_isLocal_impl(const void *);
}

static __inline__ __attribute__((device)) unsigned int __isGlobal(const void *ptr)
{
  return __nv_isGlobal_impl(ptr);
}

static __inline__ __attribute__((device)) unsigned int __isShared(const void *ptr)
{
  return __nv_isShared_impl(ptr);
}

static __inline__ __attribute__((device)) unsigned int __isConstant(const void *ptr)
{
  return __nv_isConstant_impl(ptr);
}

static __inline__ __attribute__((device)) unsigned int __isLocal(const void *ptr)
{
  return __nv_isLocal_impl(ptr);
}

extern "C" {
  __attribute__((device)) size_t __nv_cvta_generic_to_global_impl(const void *);
  __attribute__((device)) size_t __nv_cvta_generic_to_shared_impl(const void *);
  __attribute__((device)) size_t __nv_cvta_generic_to_constant_impl(const void *);
  __attribute__((device)) size_t __nv_cvta_generic_to_local_impl(const void *);
  __attribute__((device)) void * __nv_cvta_global_to_generic_impl(size_t);
  __attribute__((device)) void * __nv_cvta_shared_to_generic_impl(size_t);
  __attribute__((device)) void * __nv_cvta_constant_to_generic_impl(size_t);
  __attribute__((device)) void * __nv_cvta_local_to_generic_impl(size_t);
}

static __inline__ __attribute__((device)) size_t __cvta_generic_to_global(const void *p)
{
  return __nv_cvta_generic_to_global_impl(p);
}

static __inline__ __attribute__((device)) size_t __cvta_generic_to_shared(const void *p)
{
  return __nv_cvta_generic_to_shared_impl(p);
}

static __inline__ __attribute__((device)) size_t __cvta_generic_to_constant(const void *p)
{
  return __nv_cvta_generic_to_constant_impl(p);
}

static __inline__ __attribute__((device)) size_t __cvta_generic_to_local(const void *p)
{
  return __nv_cvta_generic_to_local_impl(p);
}

static __inline__ __attribute__((device)) void * __cvta_global_to_generic(size_t rawbits)
{
  return __nv_cvta_global_to_generic_impl(rawbits);
}

static __inline__ __attribute__((device)) void * __cvta_shared_to_generic(size_t rawbits)
{
  return __nv_cvta_shared_to_generic_impl(rawbits);
}

static __inline__ __attribute__((device)) void * __cvta_constant_to_generic(size_t rawbits)
{
  return __nv_cvta_constant_to_generic_impl(rawbits);
}

static __inline__ __attribute__((device)) void * __cvta_local_to_generic(size_t rawbits)
{
  return __nv_cvta_local_to_generic_impl(rawbits);
}
# 1543 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h" 2
# 3285 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h" 1
# 102 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
static __attribute__((device)) __inline__ unsigned __fns(unsigned mask, unsigned base, int offset) ;
static __attribute__((device)) __inline__ void __barrier_sync(unsigned id) ;
static __attribute__((device)) __inline__ void __barrier_sync_count(unsigned id, unsigned cnt) ;
static __attribute__((device)) __inline__ void __syncwarp(unsigned mask=0xFFFFFFFF) ;
static __attribute__((device)) __inline__ int __all_sync(unsigned mask, int pred) ;
static __attribute__((device)) __inline__ int __any_sync(unsigned mask, int pred) ;
static __attribute__((device)) __inline__ int __uni_sync(unsigned mask, int pred) ;
static __attribute__((device)) __inline__ unsigned __ballot_sync(unsigned mask, int pred) ;
static __attribute__((device)) __inline__ unsigned __activemask() ;
# 119 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is deprecated in favor of ""__shfl""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) int __shfl(int var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is deprecated in favor of ""__shfl""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned int __shfl(unsigned int var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is deprecated in favor of ""__shfl_up""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) int __shfl_up(int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is deprecated in favor of ""__shfl_up""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned int __shfl_up(unsigned int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is deprecated in favor of ""__shfl_down""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) int __shfl_down(int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is deprecated in favor of ""__shfl_down""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned int __shfl_down(unsigned int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is deprecated in favor of ""__shfl_xor""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) int __shfl_xor(int var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is deprecated in favor of ""__shfl_xor""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned int __shfl_xor(unsigned int var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is deprecated in favor of ""__shfl""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) float __shfl(float var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is deprecated in favor of ""__shfl_up""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) float __shfl_up(float var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is deprecated in favor of ""__shfl_down""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) float __shfl_down(float var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is deprecated in favor of ""__shfl_xor""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) float __shfl_xor(float var, int laneMask, int width=32) ;


static __attribute__((device)) __inline__ int __shfl_sync(unsigned mask, int var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ unsigned int __shfl_sync(unsigned mask, unsigned int var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ int __shfl_up_sync(unsigned mask, int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned int __shfl_up_sync(unsigned mask, unsigned int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ int __shfl_down_sync(unsigned mask, int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned int __shfl_down_sync(unsigned mask, unsigned int var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ unsigned int __shfl_xor_sync(unsigned mask, unsigned int var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ float __shfl_sync(unsigned mask, float var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ float __shfl_up_sync(unsigned mask, float var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ float __shfl_down_sync(unsigned mask, float var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width=32) ;



static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is deprecated in favor of ""__shfl""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned long long __shfl(unsigned long long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is deprecated in favor of ""__shfl""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) long long __shfl(long long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is deprecated in favor of ""__shfl_up""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) long long __shfl_up(long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is deprecated in favor of ""__shfl_up""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned long long __shfl_up(unsigned long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is deprecated in favor of ""__shfl_down""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) long long __shfl_down(long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is deprecated in favor of ""__shfl_down""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned long long __shfl_down(unsigned long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is deprecated in favor of ""__shfl_xor""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) long long __shfl_xor(long long var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is deprecated in favor of ""__shfl_xor""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is deprecated in favor of ""__shfl""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) double __shfl(double var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is deprecated in favor of ""__shfl_up""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) double __shfl_up(double var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is deprecated in favor of ""__shfl_down""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) double __shfl_down(double var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is deprecated in favor of ""__shfl_xor""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) double __shfl_xor(double var, int laneMask, int width=32) ;


static __attribute__((device)) __inline__ long long __shfl_sync(unsigned mask, long long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ long long __shfl_up_sync(unsigned mask, long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ long long __shfl_down_sync(unsigned mask, long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ double __shfl_sync(unsigned mask, double var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ double __shfl_up_sync(unsigned mask, double var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ double __shfl_down_sync(unsigned mask, double var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width=32) ;



static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is deprecated in favor of ""__shfl""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) long __shfl(long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl""() is deprecated in favor of ""__shfl""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned long __shfl(unsigned long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is deprecated in favor of ""__shfl_up""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) long __shfl_up(long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_up""() is deprecated in favor of ""__shfl_up""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned long __shfl_up(unsigned long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is deprecated in favor of ""__shfl_down""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) long __shfl_down(long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_down""() is deprecated in favor of ""__shfl_down""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned long __shfl_down(unsigned long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is deprecated in favor of ""__shfl_xor""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) long __shfl_xor(long var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ __attribute__((deprecated("__shfl_xor""() is deprecated in favor of ""__shfl_xor""_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."))) unsigned long __shfl_xor(unsigned long var, int laneMask, int width=32) ;


static __attribute__((device)) __inline__ long __shfl_sync(unsigned mask, long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width=32) ;
static __attribute__((device)) __inline__ long __shfl_up_sync(unsigned mask, long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ long __shfl_down_sync(unsigned mask, long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned int delta, int width=32) ;
static __attribute__((device)) __inline__ long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width=32) ;
static __attribute__((device)) __inline__ unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width=32) ;
# 212 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp" 1
# 73 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp"
extern "C"
{
}
# 89 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp"
static __attribute__((device)) __inline__
unsigned __fns(unsigned mask, unsigned base, int offset) {
  extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __nvvm_fns(unsigned int mask, unsigned int base, int offset);
  return __nvvm_fns(mask, base, offset);
}

static __attribute__((device)) __inline__
void __barrier_sync(unsigned id) {
  extern __attribute__((device)) __attribute__((device_builtin)) void __nvvm_barrier_sync(unsigned id);
  return __nvvm_barrier_sync(id);
}

static __attribute__((device)) __inline__
void __barrier_sync_count(unsigned id, unsigned cnt) {
  extern __attribute__((device)) __attribute__((device_builtin)) void __nvvm_barrier_sync_cnt(unsigned id, unsigned cnt);
  return __nvvm_barrier_sync_cnt(id, cnt);
}

static __attribute__((device)) __inline__
void __syncwarp(unsigned mask) {
  extern __attribute__((device)) __attribute__((device_builtin)) void __nvvm_bar_warp_sync(unsigned mask);
  return __nvvm_bar_warp_sync(mask);
}

static __attribute__((device)) __inline__
int __all_sync(unsigned mask, int pred) {
  extern __attribute__((device)) __attribute__((device_builtin)) int __nvvm_vote_all_sync(unsigned int mask, int pred);
  return __nvvm_vote_all_sync(mask, pred);
}

static __attribute__((device)) __inline__
int __any_sync(unsigned mask, int pred) {
  extern __attribute__((device)) __attribute__((device_builtin)) int __nvvm_vote_any_sync(unsigned int mask, int pred);
  return __nvvm_vote_any_sync(mask, pred);
}

static __attribute__((device)) __inline__
int __uni_sync(unsigned mask, int pred) {
  extern __attribute__((device)) __attribute__((device_builtin)) int __nvvm_vote_uni_sync(unsigned int mask, int pred);
  return __nvvm_vote_uni_sync(mask, pred);
}

static __attribute__((device)) __inline__
unsigned __ballot_sync(unsigned mask, int pred) {
  extern __attribute__((device)) __attribute__((device_builtin)) unsigned int __nvvm_vote_ballot_sync(unsigned int mask, int pred);
  return __nvvm_vote_ballot_sync(mask, pred);
}

static __attribute__((device)) __inline__
unsigned __activemask() {
    unsigned ret;
    asm volatile ("activemask.b32 %0;" : "=r"(ret));
    return ret;
}




static __attribute__((device)) __inline__ int __shfl(int var, int srcLane, int width) {
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.idx.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(srcLane), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl(unsigned int var, int srcLane, int width) {
 return (unsigned int) __shfl((int)var, srcLane, width);
}

static __attribute__((device)) __inline__ int __shfl_up(int var, unsigned int delta, int width) {
 int ret;
 int c = (32 -width) << 8;
 asm volatile ("shfl.up.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(delta), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl_up(unsigned int var, unsigned int delta, int width) {
 return (unsigned int) __shfl_up((int)var, delta, width);
}

static __attribute__((device)) __inline__ int __shfl_down(int var, unsigned int delta, int width) {
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.down.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(delta), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl_down(unsigned int var, unsigned int delta, int width) {
 return (unsigned int) __shfl_down((int)var, delta, width);
}

static __attribute__((device)) __inline__ int __shfl_xor(int var, int laneMask, int width) {
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.bfly.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(laneMask), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl_xor(unsigned int var, int laneMask, int width) {
 return (unsigned int) __shfl_xor((int)var, laneMask, width);
}

static __attribute__((device)) __inline__ float __shfl(float var, int srcLane, int width) {
 float ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.idx.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(srcLane), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ float __shfl_up(float var, unsigned int delta, int width) {
 float ret;
        int c;
 c = (32 -width) << 8;
 asm volatile ("shfl.up.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(delta), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ float __shfl_down(float var, unsigned int delta, int width) {
 float ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.down.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(delta), "r"(c));
 return ret;
}

static __attribute__((device)) __inline__ float __shfl_xor(float var, int laneMask, int width) {
 float ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
 asm volatile ("shfl.bfly.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(laneMask), "r"(c));
 return ret;
}



static __attribute__((device)) __inline__ long long __shfl(long long var, int srcLane, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl(hi, srcLane, width);
 lo = __shfl(lo, srcLane, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl(unsigned long long var, int srcLane, int width) {
 return (unsigned long long) __shfl((long long) var, srcLane, width);
}

static __attribute__((device)) __inline__ long long __shfl_up(long long var, unsigned int delta, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_up(hi, delta, width);
 lo = __shfl_up(lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl_up(unsigned long long var, unsigned int delta, int width) {
 return (unsigned long long) __shfl_up((long long) var, delta, width);
}

static __attribute__((device)) __inline__ long long __shfl_down(long long var, unsigned int delta, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_down(hi, delta, width);
 lo = __shfl_down(lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl_down(unsigned long long var, unsigned int delta, int width) {
 return (unsigned long long) __shfl_down((long long) var, delta, width);
}

static __attribute__((device)) __inline__ long long __shfl_xor(long long var, int laneMask, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_xor(hi, laneMask, width);
 lo = __shfl_xor(lo, laneMask, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width) {
 return (unsigned long long) __shfl_xor((long long) var, laneMask, width);
}

static __attribute__((device)) __inline__ double __shfl(double var, int srcLane, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl(hi, srcLane, width);
 lo = __shfl(lo, srcLane, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_up(double var, unsigned int delta, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_up(hi, delta, width);
 lo = __shfl_up(lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_down(double var, unsigned int delta, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_down(hi, delta, width);
 lo = __shfl_down(lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_xor(double var, int laneMask, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_xor(hi, laneMask, width);
 lo = __shfl_xor(lo, laneMask, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ long __shfl(long var, int srcLane, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl((long long) var, srcLane, width) :
  __shfl((int) var, srcLane, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl(unsigned long var, int srcLane, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl((unsigned long long) var, srcLane, width) :
  __shfl((unsigned int) var, srcLane, width);
}

static __attribute__((device)) __inline__ long __shfl_up(long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_up((long long) var, delta, width) :
  __shfl_up((int) var, delta, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_up(unsigned long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_up((unsigned long long) var, delta, width) :
  __shfl_up((unsigned int) var, delta, width);
}

static __attribute__((device)) __inline__ long __shfl_down(long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_down((long long) var, delta, width) :
  __shfl_down((int) var, delta, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_down(unsigned long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_down((unsigned long long) var, delta, width) :
  __shfl_down((unsigned int) var, delta, width);
}

static __attribute__((device)) __inline__ long __shfl_xor(long var, int laneMask, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_xor((long long) var, laneMask, width) :
  __shfl_xor((int) var, laneMask, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_xor(unsigned long var, int laneMask, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_xor((unsigned long long) var, laneMask, width) :
  __shfl_xor((unsigned int) var, laneMask, width);
}
# 369 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.hpp"
static __attribute__((device)) __inline__ int __shfl_sync(unsigned mask, int var, int srcLane, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_idx_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_idx_sync(mask, var, srcLane, c);
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl_sync(unsigned mask, unsigned int var, int srcLane, int width) {
        return (unsigned int) __shfl_sync(mask, (int)var, srcLane, width);
}

static __attribute__((device)) __inline__ int __shfl_up_sync(unsigned mask, int var, unsigned int delta, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_up_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
 int c = (32 -width) << 8;
        ret = __nvvm_shfl_up_sync(mask, var, delta, c);
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl_up_sync(unsigned mask, unsigned int var, unsigned int delta, int width) {
        return (unsigned int) __shfl_up_sync(mask, (int)var, delta, width);
}

static __attribute__((device)) __inline__ int __shfl_down_sync(unsigned mask, int var, unsigned int delta, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_down_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_down_sync(mask, var, delta, c);
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl_down_sync(unsigned mask, unsigned int var, unsigned int delta, int width) {
        return (unsigned int) __shfl_down_sync(mask, (int)var, delta, width);
}

static __attribute__((device)) __inline__ int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_bfly_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
 int c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_bfly_sync(mask, var, laneMask, c);
 return ret;
}

static __attribute__((device)) __inline__ unsigned int __shfl_xor_sync(unsigned mask, unsigned int var, int laneMask, int width) {
 return (unsigned int) __shfl_xor_sync(mask, (int)var, laneMask, width);
}

static __attribute__((device)) __inline__ float __shfl_sync(unsigned mask, float var, int srcLane, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_idx_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
        int ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_idx_sync(mask, __float_as_int(var), srcLane, c);
 return __int_as_float(ret);
}

static __attribute__((device)) __inline__ float __shfl_up_sync(unsigned mask, float var, unsigned int delta, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_up_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
        int c;
 c = (32 -width) << 8;
        ret = __nvvm_shfl_up_sync(mask, __float_as_int(var), delta, c);
 return __int_as_float(ret);
}

static __attribute__((device)) __inline__ float __shfl_down_sync(unsigned mask, float var, unsigned int delta, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_down_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_down_sync(mask, __float_as_int(var), delta, c);
 return __int_as_float(ret);
}

static __attribute__((device)) __inline__ float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width) {
        extern __attribute__((device)) __attribute__((device_builtin)) unsigned __nvvm_shfl_bfly_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
 int ret;
        int c;
 c = ((32 -width) << 8) | 0x1f;
        ret = __nvvm_shfl_bfly_sync(mask, __float_as_int(var), laneMask, c);
 return __int_as_float(ret);
}


static __attribute__((device)) __inline__ long long __shfl_sync(unsigned mask, long long var, int srcLane, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_sync(mask, hi, srcLane, width);
 lo = __shfl_sync(mask, lo, srcLane, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width) {
        return (unsigned long long) __shfl_sync(mask, (long long) var, srcLane, width);
}

static __attribute__((device)) __inline__ long long __shfl_up_sync(unsigned mask, long long var, unsigned int delta, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_up_sync(mask, hi, delta, width);
 lo = __shfl_up_sync(mask, lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned int delta, int width) {
        return (unsigned long long) __shfl_up_sync(mask, (long long) var, delta, width);
}

static __attribute__((device)) __inline__ long long __shfl_down_sync(unsigned mask, long long var, unsigned int delta, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_down_sync(mask, hi, delta, width);
 lo = __shfl_down_sync(mask, lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned int delta, int width) {
        return (unsigned long long) __shfl_down_sync(mask, (long long) var, delta, width);
}

static __attribute__((device)) __inline__ long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width) {
 int lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
 hi = __shfl_xor_sync(mask, hi, laneMask, width);
 lo = __shfl_xor_sync(mask, lo, laneMask, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width) {
        return (unsigned long long) __shfl_xor_sync(mask, (long long) var, laneMask, width);
}

static __attribute__((device)) __inline__ double __shfl_sync(unsigned mask, double var, int srcLane, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_sync(mask, hi, srcLane, width);
 lo = __shfl_sync(mask, lo, srcLane, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_up_sync(unsigned mask, double var, unsigned int delta, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_up_sync(mask, hi, delta, width);
 lo = __shfl_up_sync(mask, lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_down_sync(unsigned mask, double var, unsigned int delta, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_down_sync(mask, hi, delta, width);
 lo = __shfl_down_sync(mask, lo, delta, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}

static __attribute__((device)) __inline__ double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width) {
 unsigned lo, hi;
 asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
 hi = __shfl_xor_sync(mask, hi, laneMask, width);
 lo = __shfl_xor_sync(mask, lo, laneMask, width);
 asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
 return var;
}



static __attribute__((device)) __inline__ long __shfl_sync(unsigned mask, long var, int srcLane, int width) {
 return (sizeof(long) == sizeof(long long)) ?
                __shfl_sync(mask, (long long) var, srcLane, width) :
  __shfl_sync(mask, (int) var, srcLane, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width) {
 return (sizeof(long) == sizeof(long long)) ?
                __shfl_sync(mask, (unsigned long long) var, srcLane, width) :
  __shfl_sync(mask, (unsigned int) var, srcLane, width);
}

static __attribute__((device)) __inline__ long __shfl_up_sync(unsigned mask, long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_up_sync(mask, (long long) var, delta, width) :
  __shfl_up_sync(mask, (int) var, delta, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_up_sync(mask, (unsigned long long) var, delta, width) :
  __shfl_up_sync(mask, (unsigned int) var, delta, width);
}

static __attribute__((device)) __inline__ long __shfl_down_sync(unsigned mask, long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_down_sync(mask, (long long) var, delta, width) :
  __shfl_down_sync(mask, (int) var, delta, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned int delta, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_down_sync(mask, (unsigned long long) var, delta, width) :
  __shfl_down_sync(mask, (unsigned int) var, delta, width);
}

static __attribute__((device)) __inline__ long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_xor_sync(mask, (long long) var, laneMask, width) :
  __shfl_xor_sync(mask, (int) var, laneMask, width);
}

static __attribute__((device)) __inline__ unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width) {
 return (sizeof(long) == sizeof(long long)) ?
  __shfl_xor_sync(mask, (unsigned long long) var, laneMask, width) :
  __shfl_xor_sync(mask, (unsigned int) var, laneMask, width);
}
# 213 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h" 2
# 3286 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h" 1
# 87 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
static __attribute__((device)) __inline__ long __ldg(const long *ptr) ;
static __attribute__((device)) __inline__ unsigned long __ldg(const unsigned long *ptr) ;

static __attribute__((device)) __inline__ char __ldg(const char *ptr) ;
static __attribute__((device)) __inline__ signed char __ldg(const signed char *ptr) ;
static __attribute__((device)) __inline__ short __ldg(const short *ptr) ;
static __attribute__((device)) __inline__ int __ldg(const int *ptr) ;
static __attribute__((device)) __inline__ long long __ldg(const long long *ptr) ;
static __attribute__((device)) __inline__ char2 __ldg(const char2 *ptr) ;
static __attribute__((device)) __inline__ char4 __ldg(const char4 *ptr) ;
static __attribute__((device)) __inline__ short2 __ldg(const short2 *ptr) ;
static __attribute__((device)) __inline__ short4 __ldg(const short4 *ptr) ;
static __attribute__((device)) __inline__ int2 __ldg(const int2 *ptr) ;
static __attribute__((device)) __inline__ int4 __ldg(const int4 *ptr) ;
static __attribute__((device)) __inline__ longlong2 __ldg(const longlong2 *ptr) ;

static __attribute__((device)) __inline__ unsigned char __ldg(const unsigned char *ptr) ;
static __attribute__((device)) __inline__ unsigned short __ldg(const unsigned short *ptr) ;
static __attribute__((device)) __inline__ unsigned int __ldg(const unsigned int *ptr) ;
static __attribute__((device)) __inline__ unsigned long long __ldg(const unsigned long long *ptr) ;
static __attribute__((device)) __inline__ uchar2 __ldg(const uchar2 *ptr) ;
static __attribute__((device)) __inline__ uchar4 __ldg(const uchar4 *ptr) ;
static __attribute__((device)) __inline__ ushort2 __ldg(const ushort2 *ptr) ;
static __attribute__((device)) __inline__ ushort4 __ldg(const ushort4 *ptr) ;
static __attribute__((device)) __inline__ uint2 __ldg(const uint2 *ptr) ;
static __attribute__((device)) __inline__ uint4 __ldg(const uint4 *ptr) ;
static __attribute__((device)) __inline__ ulonglong2 __ldg(const ulonglong2 *ptr) ;

static __attribute__((device)) __inline__ float __ldg(const float *ptr) ;
static __attribute__((device)) __inline__ double __ldg(const double *ptr) ;
static __attribute__((device)) __inline__ float2 __ldg(const float2 *ptr) ;
static __attribute__((device)) __inline__ float4 __ldg(const float4 *ptr) ;
static __attribute__((device)) __inline__ double2 __ldg(const double2 *ptr) ;



static __attribute__((device)) __inline__ long __ldcg(const long *ptr) ;
static __attribute__((device)) __inline__ unsigned long __ldcg(const unsigned long *ptr) ;

static __attribute__((device)) __inline__ char __ldcg(const char *ptr) ;
static __attribute__((device)) __inline__ signed char __ldcg(const signed char *ptr) ;
static __attribute__((device)) __inline__ short __ldcg(const short *ptr) ;
static __attribute__((device)) __inline__ int __ldcg(const int *ptr) ;
static __attribute__((device)) __inline__ long long __ldcg(const long long *ptr) ;
static __attribute__((device)) __inline__ char2 __ldcg(const char2 *ptr) ;
static __attribute__((device)) __inline__ char4 __ldcg(const char4 *ptr) ;
static __attribute__((device)) __inline__ short2 __ldcg(const short2 *ptr) ;
static __attribute__((device)) __inline__ short4 __ldcg(const short4 *ptr) ;
static __attribute__((device)) __inline__ int2 __ldcg(const int2 *ptr) ;
static __attribute__((device)) __inline__ int4 __ldcg(const int4 *ptr) ;
static __attribute__((device)) __inline__ longlong2 __ldcg(const longlong2 *ptr) ;

static __attribute__((device)) __inline__ unsigned char __ldcg(const unsigned char *ptr) ;
static __attribute__((device)) __inline__ unsigned short __ldcg(const unsigned short *ptr) ;
static __attribute__((device)) __inline__ unsigned int __ldcg(const unsigned int *ptr) ;
static __attribute__((device)) __inline__ unsigned long long __ldcg(const unsigned long long *ptr) ;
static __attribute__((device)) __inline__ uchar2 __ldcg(const uchar2 *ptr) ;
static __attribute__((device)) __inline__ uchar4 __ldcg(const uchar4 *ptr) ;
static __attribute__((device)) __inline__ ushort2 __ldcg(const ushort2 *ptr) ;
static __attribute__((device)) __inline__ ushort4 __ldcg(const ushort4 *ptr) ;
static __attribute__((device)) __inline__ uint2 __ldcg(const uint2 *ptr) ;
static __attribute__((device)) __inline__ uint4 __ldcg(const uint4 *ptr) ;
static __attribute__((device)) __inline__ ulonglong2 __ldcg(const ulonglong2 *ptr) ;

static __attribute__((device)) __inline__ float __ldcg(const float *ptr) ;
static __attribute__((device)) __inline__ double __ldcg(const double *ptr) ;
static __attribute__((device)) __inline__ float2 __ldcg(const float2 *ptr) ;
static __attribute__((device)) __inline__ float4 __ldcg(const float4 *ptr) ;
static __attribute__((device)) __inline__ double2 __ldcg(const double2 *ptr) ;



static __attribute__((device)) __inline__ long __ldca(const long *ptr) ;
static __attribute__((device)) __inline__ unsigned long __ldca(const unsigned long *ptr) ;

static __attribute__((device)) __inline__ char __ldca(const char *ptr) ;
static __attribute__((device)) __inline__ signed char __ldca(const signed char *ptr) ;
static __attribute__((device)) __inline__ short __ldca(const short *ptr) ;
static __attribute__((device)) __inline__ int __ldca(const int *ptr) ;
static __attribute__((device)) __inline__ long long __ldca(const long long *ptr) ;
static __attribute__((device)) __inline__ char2 __ldca(const char2 *ptr) ;
static __attribute__((device)) __inline__ char4 __ldca(const char4 *ptr) ;
static __attribute__((device)) __inline__ short2 __ldca(const short2 *ptr) ;
static __attribute__((device)) __inline__ short4 __ldca(const short4 *ptr) ;
static __attribute__((device)) __inline__ int2 __ldca(const int2 *ptr) ;
static __attribute__((device)) __inline__ int4 __ldca(const int4 *ptr) ;
static __attribute__((device)) __inline__ longlong2 __ldca(const longlong2 *ptr) ;

static __attribute__((device)) __inline__ unsigned char __ldca(const unsigned char *ptr) ;
static __attribute__((device)) __inline__ unsigned short __ldca(const unsigned short *ptr) ;
static __attribute__((device)) __inline__ unsigned int __ldca(const unsigned int *ptr) ;
static __attribute__((device)) __inline__ unsigned long long __ldca(const unsigned long long *ptr) ;
static __attribute__((device)) __inline__ uchar2 __ldca(const uchar2 *ptr) ;
static __attribute__((device)) __inline__ uchar4 __ldca(const uchar4 *ptr) ;
static __attribute__((device)) __inline__ ushort2 __ldca(const ushort2 *ptr) ;
static __attribute__((device)) __inline__ ushort4 __ldca(const ushort4 *ptr) ;
static __attribute__((device)) __inline__ uint2 __ldca(const uint2 *ptr) ;
static __attribute__((device)) __inline__ uint4 __ldca(const uint4 *ptr) ;
static __attribute__((device)) __inline__ ulonglong2 __ldca(const ulonglong2 *ptr) ;

static __attribute__((device)) __inline__ float __ldca(const float *ptr) ;
static __attribute__((device)) __inline__ double __ldca(const double *ptr) ;
static __attribute__((device)) __inline__ float2 __ldca(const float2 *ptr) ;
static __attribute__((device)) __inline__ float4 __ldca(const float4 *ptr) ;
static __attribute__((device)) __inline__ double2 __ldca(const double2 *ptr) ;



static __attribute__((device)) __inline__ long __ldcs(const long *ptr) ;
static __attribute__((device)) __inline__ unsigned long __ldcs(const unsigned long *ptr) ;

static __attribute__((device)) __inline__ char __ldcs(const char *ptr) ;
static __attribute__((device)) __inline__ signed char __ldcs(const signed char *ptr) ;
static __attribute__((device)) __inline__ short __ldcs(const short *ptr) ;
static __attribute__((device)) __inline__ int __ldcs(const int *ptr) ;
static __attribute__((device)) __inline__ long long __ldcs(const long long *ptr) ;
static __attribute__((device)) __inline__ char2 __ldcs(const char2 *ptr) ;
static __attribute__((device)) __inline__ char4 __ldcs(const char4 *ptr) ;
static __attribute__((device)) __inline__ short2 __ldcs(const short2 *ptr) ;
static __attribute__((device)) __inline__ short4 __ldcs(const short4 *ptr) ;
static __attribute__((device)) __inline__ int2 __ldcs(const int2 *ptr) ;
static __attribute__((device)) __inline__ int4 __ldcs(const int4 *ptr) ;
static __attribute__((device)) __inline__ longlong2 __ldcs(const longlong2 *ptr) ;

static __attribute__((device)) __inline__ unsigned char __ldcs(const unsigned char *ptr) ;
static __attribute__((device)) __inline__ unsigned short __ldcs(const unsigned short *ptr) ;
static __attribute__((device)) __inline__ unsigned int __ldcs(const unsigned int *ptr) ;
static __attribute__((device)) __inline__ unsigned long long __ldcs(const unsigned long long *ptr) ;
static __attribute__((device)) __inline__ uchar2 __ldcs(const uchar2 *ptr) ;
static __attribute__((device)) __inline__ uchar4 __ldcs(const uchar4 *ptr) ;
static __attribute__((device)) __inline__ ushort2 __ldcs(const ushort2 *ptr) ;
static __attribute__((device)) __inline__ ushort4 __ldcs(const ushort4 *ptr) ;
static __attribute__((device)) __inline__ uint2 __ldcs(const uint2 *ptr) ;
static __attribute__((device)) __inline__ uint4 __ldcs(const uint4 *ptr) ;
static __attribute__((device)) __inline__ ulonglong2 __ldcs(const ulonglong2 *ptr) ;

static __attribute__((device)) __inline__ float __ldcs(const float *ptr) ;
static __attribute__((device)) __inline__ double __ldcs(const double *ptr) ;
static __attribute__((device)) __inline__ float2 __ldcs(const float2 *ptr) ;
static __attribute__((device)) __inline__ float4 __ldcs(const float4 *ptr) ;
static __attribute__((device)) __inline__ double2 __ldcs(const double2 *ptr) ;



static __attribute__((device)) __inline__ long __ldlu(const long *ptr) ;
static __attribute__((device)) __inline__ unsigned long __ldlu(const unsigned long *ptr) ;

static __attribute__((device)) __inline__ char __ldlu(const char *ptr) ;
static __attribute__((device)) __inline__ signed char __ldlu(const signed char *ptr) ;
static __attribute__((device)) __inline__ short __ldlu(const short *ptr) ;
static __attribute__((device)) __inline__ int __ldlu(const int *ptr) ;
static __attribute__((device)) __inline__ long long __ldlu(const long long *ptr) ;
static __attribute__((device)) __inline__ char2 __ldlu(const char2 *ptr) ;
static __attribute__((device)) __inline__ char4 __ldlu(const char4 *ptr) ;
static __attribute__((device)) __inline__ short2 __ldlu(const short2 *ptr) ;
static __attribute__((device)) __inline__ short4 __ldlu(const short4 *ptr) ;
static __attribute__((device)) __inline__ int2 __ldlu(const int2 *ptr) ;
static __attribute__((device)) __inline__ int4 __ldlu(const int4 *ptr) ;
static __attribute__((device)) __inline__ longlong2 __ldlu(const longlong2 *ptr) ;

static __attribute__((device)) __inline__ unsigned char __ldlu(const unsigned char *ptr) ;
static __attribute__((device)) __inline__ unsigned short __ldlu(const unsigned short *ptr) ;
static __attribute__((device)) __inline__ unsigned int __ldlu(const unsigned int *ptr) ;
static __attribute__((device)) __inline__ unsigned long long __ldlu(const unsigned long long *ptr) ;
static __attribute__((device)) __inline__ uchar2 __ldlu(const uchar2 *ptr) ;
static __attribute__((device)) __inline__ uchar4 __ldlu(const uchar4 *ptr) ;
static __attribute__((device)) __inline__ ushort2 __ldlu(const ushort2 *ptr) ;
static __attribute__((device)) __inline__ ushort4 __ldlu(const ushort4 *ptr) ;
static __attribute__((device)) __inline__ uint2 __ldlu(const uint2 *ptr) ;
static __attribute__((device)) __inline__ uint4 __ldlu(const uint4 *ptr) ;
static __attribute__((device)) __inline__ ulonglong2 __ldlu(const ulonglong2 *ptr) ;

static __attribute__((device)) __inline__ float __ldlu(const float *ptr) ;
static __attribute__((device)) __inline__ double __ldlu(const double *ptr) ;
static __attribute__((device)) __inline__ float2 __ldlu(const float2 *ptr) ;
static __attribute__((device)) __inline__ float4 __ldlu(const float4 *ptr) ;
static __attribute__((device)) __inline__ double2 __ldlu(const double2 *ptr) ;



static __attribute__((device)) __inline__ long __ldcv(const long *ptr) ;
static __attribute__((device)) __inline__ unsigned long __ldcv(const unsigned long *ptr) ;

static __attribute__((device)) __inline__ char __ldcv(const char *ptr) ;
static __attribute__((device)) __inline__ signed char __ldcv(const signed char *ptr) ;
static __attribute__((device)) __inline__ short __ldcv(const short *ptr) ;
static __attribute__((device)) __inline__ int __ldcv(const int *ptr) ;
static __attribute__((device)) __inline__ long long __ldcv(const long long *ptr) ;
static __attribute__((device)) __inline__ char2 __ldcv(const char2 *ptr) ;
static __attribute__((device)) __inline__ char4 __ldcv(const char4 *ptr) ;
static __attribute__((device)) __inline__ short2 __ldcv(const short2 *ptr) ;
static __attribute__((device)) __inline__ short4 __ldcv(const short4 *ptr) ;
static __attribute__((device)) __inline__ int2 __ldcv(const int2 *ptr) ;
static __attribute__((device)) __inline__ int4 __ldcv(const int4 *ptr) ;
static __attribute__((device)) __inline__ longlong2 __ldcv(const longlong2 *ptr) ;

static __attribute__((device)) __inline__ unsigned char __ldcv(const unsigned char *ptr) ;
static __attribute__((device)) __inline__ unsigned short __ldcv(const unsigned short *ptr) ;
static __attribute__((device)) __inline__ unsigned int __ldcv(const unsigned int *ptr) ;
static __attribute__((device)) __inline__ unsigned long long __ldcv(const unsigned long long *ptr) ;
static __attribute__((device)) __inline__ uchar2 __ldcv(const uchar2 *ptr) ;
static __attribute__((device)) __inline__ uchar4 __ldcv(const uchar4 *ptr) ;
static __attribute__((device)) __inline__ ushort2 __ldcv(const ushort2 *ptr) ;
static __attribute__((device)) __inline__ ushort4 __ldcv(const ushort4 *ptr) ;
static __attribute__((device)) __inline__ uint2 __ldcv(const uint2 *ptr) ;
static __attribute__((device)) __inline__ uint4 __ldcv(const uint4 *ptr) ;
static __attribute__((device)) __inline__ ulonglong2 __ldcv(const ulonglong2 *ptr) ;

static __attribute__((device)) __inline__ float __ldcv(const float *ptr) ;
static __attribute__((device)) __inline__ double __ldcv(const double *ptr) ;
static __attribute__((device)) __inline__ float2 __ldcv(const float2 *ptr) ;
static __attribute__((device)) __inline__ float4 __ldcv(const float4 *ptr) ;
static __attribute__((device)) __inline__ double2 __ldcv(const double2 *ptr) ;



static __attribute__((device)) __inline__ void __stwb(long *ptr, long value) ;
static __attribute__((device)) __inline__ void __stwb(unsigned long *ptr, unsigned long value) ;

static __attribute__((device)) __inline__ void __stwb(char *ptr, char value) ;
static __attribute__((device)) __inline__ void __stwb(signed char *ptr, signed char value) ;
static __attribute__((device)) __inline__ void __stwb(short *ptr, short value) ;
static __attribute__((device)) __inline__ void __stwb(int *ptr, int value) ;
static __attribute__((device)) __inline__ void __stwb(long long *ptr, long long value) ;
static __attribute__((device)) __inline__ void __stwb(char2 *ptr, char2 value) ;
static __attribute__((device)) __inline__ void __stwb(char4 *ptr, char4 value) ;
static __attribute__((device)) __inline__ void __stwb(short2 *ptr, short2 value) ;
static __attribute__((device)) __inline__ void __stwb(short4 *ptr, short4 value) ;
static __attribute__((device)) __inline__ void __stwb(int2 *ptr, int2 value) ;
static __attribute__((device)) __inline__ void __stwb(int4 *ptr, int4 value) ;
static __attribute__((device)) __inline__ void __stwb(longlong2 *ptr, longlong2 value) ;

static __attribute__((device)) __inline__ void __stwb(unsigned char *ptr, unsigned char value) ;
static __attribute__((device)) __inline__ void __stwb(unsigned short *ptr, unsigned short value) ;
static __attribute__((device)) __inline__ void __stwb(unsigned int *ptr, unsigned int value) ;
static __attribute__((device)) __inline__ void __stwb(unsigned long long *ptr, unsigned long long value) ;
static __attribute__((device)) __inline__ void __stwb(uchar2 *ptr, uchar2 value) ;
static __attribute__((device)) __inline__ void __stwb(uchar4 *ptr, uchar4 value) ;
static __attribute__((device)) __inline__ void __stwb(ushort2 *ptr, ushort2 value) ;
static __attribute__((device)) __inline__ void __stwb(ushort4 *ptr, ushort4 value) ;
static __attribute__((device)) __inline__ void __stwb(uint2 *ptr, uint2 value) ;
static __attribute__((device)) __inline__ void __stwb(uint4 *ptr, uint4 value) ;
static __attribute__((device)) __inline__ void __stwb(ulonglong2 *ptr, ulonglong2 value) ;

static __attribute__((device)) __inline__ void __stwb(float *ptr, float value) ;
static __attribute__((device)) __inline__ void __stwb(double *ptr, double value) ;
static __attribute__((device)) __inline__ void __stwb(float2 *ptr, float2 value) ;
static __attribute__((device)) __inline__ void __stwb(float4 *ptr, float4 value) ;
static __attribute__((device)) __inline__ void __stwb(double2 *ptr, double2 value) ;



static __attribute__((device)) __inline__ void __stcg(long *ptr, long value) ;
static __attribute__((device)) __inline__ void __stcg(unsigned long *ptr, unsigned long value) ;

static __attribute__((device)) __inline__ void __stcg(char *ptr, char value) ;
static __attribute__((device)) __inline__ void __stcg(signed char *ptr, signed char value) ;
static __attribute__((device)) __inline__ void __stcg(short *ptr, short value) ;
static __attribute__((device)) __inline__ void __stcg(int *ptr, int value) ;
static __attribute__((device)) __inline__ void __stcg(long long *ptr, long long value) ;
static __attribute__((device)) __inline__ void __stcg(char2 *ptr, char2 value) ;
static __attribute__((device)) __inline__ void __stcg(char4 *ptr, char4 value) ;
static __attribute__((device)) __inline__ void __stcg(short2 *ptr, short2 value) ;
static __attribute__((device)) __inline__ void __stcg(short4 *ptr, short4 value) ;
static __attribute__((device)) __inline__ void __stcg(int2 *ptr, int2 value) ;
static __attribute__((device)) __inline__ void __stcg(int4 *ptr, int4 value) ;
static __attribute__((device)) __inline__ void __stcg(longlong2 *ptr, longlong2 value) ;

static __attribute__((device)) __inline__ void __stcg(unsigned char *ptr, unsigned char value) ;
static __attribute__((device)) __inline__ void __stcg(unsigned short *ptr, unsigned short value) ;
static __attribute__((device)) __inline__ void __stcg(unsigned int *ptr, unsigned int value) ;
static __attribute__((device)) __inline__ void __stcg(unsigned long long *ptr, unsigned long long value) ;
static __attribute__((device)) __inline__ void __stcg(uchar2 *ptr, uchar2 value) ;
static __attribute__((device)) __inline__ void __stcg(uchar4 *ptr, uchar4 value) ;
static __attribute__((device)) __inline__ void __stcg(ushort2 *ptr, ushort2 value) ;
static __attribute__((device)) __inline__ void __stcg(ushort4 *ptr, ushort4 value) ;
static __attribute__((device)) __inline__ void __stcg(uint2 *ptr, uint2 value) ;
static __attribute__((device)) __inline__ void __stcg(uint4 *ptr, uint4 value) ;
static __attribute__((device)) __inline__ void __stcg(ulonglong2 *ptr, ulonglong2 value) ;

static __attribute__((device)) __inline__ void __stcg(float *ptr, float value) ;
static __attribute__((device)) __inline__ void __stcg(double *ptr, double value) ;
static __attribute__((device)) __inline__ void __stcg(float2 *ptr, float2 value) ;
static __attribute__((device)) __inline__ void __stcg(float4 *ptr, float4 value) ;
static __attribute__((device)) __inline__ void __stcg(double2 *ptr, double2 value) ;



static __attribute__((device)) __inline__ void __stcs(long *ptr, long value) ;
static __attribute__((device)) __inline__ void __stcs(unsigned long *ptr, unsigned long value) ;

static __attribute__((device)) __inline__ void __stcs(char *ptr, char value) ;
static __attribute__((device)) __inline__ void __stcs(signed char *ptr, signed char value) ;
static __attribute__((device)) __inline__ void __stcs(short *ptr, short value) ;
static __attribute__((device)) __inline__ void __stcs(int *ptr, int value) ;
static __attribute__((device)) __inline__ void __stcs(long long *ptr, long long value) ;
static __attribute__((device)) __inline__ void __stcs(char2 *ptr, char2 value) ;
static __attribute__((device)) __inline__ void __stcs(char4 *ptr, char4 value) ;
static __attribute__((device)) __inline__ void __stcs(short2 *ptr, short2 value) ;
static __attribute__((device)) __inline__ void __stcs(short4 *ptr, short4 value) ;
static __attribute__((device)) __inline__ void __stcs(int2 *ptr, int2 value) ;
static __attribute__((device)) __inline__ void __stcs(int4 *ptr, int4 value) ;
static __attribute__((device)) __inline__ void __stcs(longlong2 *ptr, longlong2 value) ;

static __attribute__((device)) __inline__ void __stcs(unsigned char *ptr, unsigned char value) ;
static __attribute__((device)) __inline__ void __stcs(unsigned short *ptr, unsigned short value) ;
static __attribute__((device)) __inline__ void __stcs(unsigned int *ptr, unsigned int value) ;
static __attribute__((device)) __inline__ void __stcs(unsigned long long *ptr, unsigned long long value) ;
static __attribute__((device)) __inline__ void __stcs(uchar2 *ptr, uchar2 value) ;
static __attribute__((device)) __inline__ void __stcs(uchar4 *ptr, uchar4 value) ;
static __attribute__((device)) __inline__ void __stcs(ushort2 *ptr, ushort2 value) ;
static __attribute__((device)) __inline__ void __stcs(ushort4 *ptr, ushort4 value) ;
static __attribute__((device)) __inline__ void __stcs(uint2 *ptr, uint2 value) ;
static __attribute__((device)) __inline__ void __stcs(uint4 *ptr, uint4 value) ;
static __attribute__((device)) __inline__ void __stcs(ulonglong2 *ptr, ulonglong2 value) ;

static __attribute__((device)) __inline__ void __stcs(float *ptr, float value) ;
static __attribute__((device)) __inline__ void __stcs(double *ptr, double value) ;
static __attribute__((device)) __inline__ void __stcs(float2 *ptr, float2 value) ;
static __attribute__((device)) __inline__ void __stcs(float4 *ptr, float4 value) ;
static __attribute__((device)) __inline__ void __stcs(double2 *ptr, double2 value) ;



static __attribute__((device)) __inline__ void __stwt(long *ptr, long value) ;
static __attribute__((device)) __inline__ void __stwt(unsigned long *ptr, unsigned long value) ;

static __attribute__((device)) __inline__ void __stwt(char *ptr, char value) ;
static __attribute__((device)) __inline__ void __stwt(signed char *ptr, signed char value) ;
static __attribute__((device)) __inline__ void __stwt(short *ptr, short value) ;
static __attribute__((device)) __inline__ void __stwt(int *ptr, int value) ;
static __attribute__((device)) __inline__ void __stwt(long long *ptr, long long value) ;
static __attribute__((device)) __inline__ void __stwt(char2 *ptr, char2 value) ;
static __attribute__((device)) __inline__ void __stwt(char4 *ptr, char4 value) ;
static __attribute__((device)) __inline__ void __stwt(short2 *ptr, short2 value) ;
static __attribute__((device)) __inline__ void __stwt(short4 *ptr, short4 value) ;
static __attribute__((device)) __inline__ void __stwt(int2 *ptr, int2 value) ;
static __attribute__((device)) __inline__ void __stwt(int4 *ptr, int4 value) ;
static __attribute__((device)) __inline__ void __stwt(longlong2 *ptr, longlong2 value) ;

static __attribute__((device)) __inline__ void __stwt(unsigned char *ptr, unsigned char value) ;
static __attribute__((device)) __inline__ void __stwt(unsigned short *ptr, unsigned short value) ;
static __attribute__((device)) __inline__ void __stwt(unsigned int *ptr, unsigned int value) ;
static __attribute__((device)) __inline__ void __stwt(unsigned long long *ptr, unsigned long long value) ;
static __attribute__((device)) __inline__ void __stwt(uchar2 *ptr, uchar2 value) ;
static __attribute__((device)) __inline__ void __stwt(uchar4 *ptr, uchar4 value) ;
static __attribute__((device)) __inline__ void __stwt(ushort2 *ptr, ushort2 value) ;
static __attribute__((device)) __inline__ void __stwt(ushort4 *ptr, ushort4 value) ;
static __attribute__((device)) __inline__ void __stwt(uint2 *ptr, uint2 value) ;
static __attribute__((device)) __inline__ void __stwt(uint4 *ptr, uint4 value) ;
static __attribute__((device)) __inline__ void __stwt(ulonglong2 *ptr, ulonglong2 value) ;

static __attribute__((device)) __inline__ void __stwt(float *ptr, float value) ;
static __attribute__((device)) __inline__ void __stwt(double *ptr, double value) ;
static __attribute__((device)) __inline__ void __stwt(float2 *ptr, float2 value) ;
static __attribute__((device)) __inline__ void __stwt(float4 *ptr, float4 value) ;
static __attribute__((device)) __inline__ void __stwt(double2 *ptr, double2 value) ;
# 460 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
static __attribute__((device)) __inline__ unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift) ;
# 472 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
static __attribute__((device)) __inline__ unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift) ;
# 485 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
static __attribute__((device)) __inline__ unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift) ;
# 497 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
static __attribute__((device)) __inline__ unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift) ;
# 507 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.hpp" 1
# 73 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.hpp"
extern "C"
{


}
# 101 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.hpp"
static __attribute__((device)) __inline__ long __ldg(const long *ptr) { unsigned long ret; asm volatile ("ld.global.nc.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long)ret; }
static __attribute__((device)) __inline__ unsigned long __ldg(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.nc.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }






static __attribute__((device)) __inline__ char __ldg(const char *ptr) { unsigned int ret; asm volatile ("ld.global.nc.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (char)ret; }
static __attribute__((device)) __inline__ signed char __ldg(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.nc.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (signed char)ret; }
static __attribute__((device)) __inline__ short __ldg(const short *ptr) { unsigned short ret; asm volatile ("ld.global.nc.s16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return (short)ret; }
static __attribute__((device)) __inline__ int __ldg(const int *ptr) { unsigned int ret; asm volatile ("ld.global.nc.s32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (int)ret; }
static __attribute__((device)) __inline__ long long __ldg(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.nc.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long long)ret; }
static __attribute__((device)) __inline__ char2 __ldg(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.nc.v2.s8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
static __attribute__((device)) __inline__ char4 __ldg(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.nc.v4.s8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
static __attribute__((device)) __inline__ short2 __ldg(const short2 *ptr) { short2 ret; asm volatile ("ld.global.nc.v2.s16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ short4 __ldg(const short4 *ptr) { short4 ret; asm volatile ("ld.global.nc.v4.s16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int2 __ldg(const int2 *ptr) { int2 ret; asm volatile ("ld.global.nc.v2.s32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int4 __ldg(const int4 *ptr) { int4 ret; asm volatile ("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ longlong2 __ldg(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.nc.v2.s64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ unsigned char __ldg(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.nc.u8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (unsigned char)ret; }
static __attribute__((device)) __inline__ unsigned short __ldg(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.nc.u16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned int __ldg(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.nc.u32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned long long __ldg(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.nc.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uchar2 __ldg(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.nc.v2.u8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
static __attribute__((device)) __inline__ uchar4 __ldg(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.nc.v4.u8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
static __attribute__((device)) __inline__ ushort2 __ldg(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.nc.v2.u16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ushort4 __ldg(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.nc.v4.u16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint2 __ldg(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint4 __ldg(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ulonglong2 __ldg(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.nc.v2.u64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ float __ldg(const float *ptr) { float ret; asm volatile ("ld.global.nc.f32 %0, [%1];" : "=f"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double __ldg(const double *ptr) { double ret; asm volatile ("ld.global.nc.f64 %0, [%1];" : "=d"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float2 __ldg(const float2 *ptr) { float2 ret; asm volatile ("ld.global.nc.v2.f32 {%0,%1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float4 __ldg(const float4 *ptr) { float4 ret; asm volatile ("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double2 __ldg(const double2 *ptr) { double2 ret; asm volatile ("ld.global.nc.v2.f64 {%0,%1}, [%2];" : "=d"(ret.x), "=d"(ret.y) : "l" (ptr)); return ret; }
# 147 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.hpp"
static __attribute__((device)) __inline__ long __ldcg(const long *ptr) { unsigned long ret; asm volatile ("ld.global.cg.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long)ret; }
static __attribute__((device)) __inline__ unsigned long __ldcg(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.cg.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }






static __attribute__((device)) __inline__ char __ldcg(const char *ptr) { unsigned int ret; asm volatile ("ld.global.cg.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (char)ret; }
static __attribute__((device)) __inline__ signed char __ldcg(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.cg.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (signed char)ret; }
static __attribute__((device)) __inline__ short __ldcg(const short *ptr) { unsigned short ret; asm volatile ("ld.global.cg.s16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return (short)ret; }
static __attribute__((device)) __inline__ int __ldcg(const int *ptr) { unsigned int ret; asm volatile ("ld.global.cg.s32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (int)ret; }
static __attribute__((device)) __inline__ long long __ldcg(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cg.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long long)ret; }
static __attribute__((device)) __inline__ char2 __ldcg(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.cg.v2.s8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
static __attribute__((device)) __inline__ char4 __ldcg(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.cg.v4.s8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
static __attribute__((device)) __inline__ short2 __ldcg(const short2 *ptr) { short2 ret; asm volatile ("ld.global.cg.v2.s16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ short4 __ldcg(const short4 *ptr) { short4 ret; asm volatile ("ld.global.cg.v4.s16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int2 __ldcg(const int2 *ptr) { int2 ret; asm volatile ("ld.global.cg.v2.s32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int4 __ldcg(const int4 *ptr) { int4 ret; asm volatile ("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ longlong2 __ldcg(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.cg.v2.s64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ unsigned char __ldcg(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.cg.u8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (unsigned char)ret; }
static __attribute__((device)) __inline__ unsigned short __ldcg(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.cg.u16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned int __ldcg(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned long long __ldcg(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cg.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uchar2 __ldcg(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.cg.v2.u8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
static __attribute__((device)) __inline__ uchar4 __ldcg(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.cg.v4.u8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
static __attribute__((device)) __inline__ ushort2 __ldcg(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.cg.v2.u16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ushort4 __ldcg(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.cg.v4.u16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint2 __ldcg(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint4 __ldcg(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ulonglong2 __ldcg(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.cg.v2.u64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ float __ldcg(const float *ptr) { float ret; asm volatile ("ld.global.cg.f32 %0, [%1];" : "=f"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double __ldcg(const double *ptr) { double ret; asm volatile ("ld.global.cg.f64 %0, [%1];" : "=d"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float2 __ldcg(const float2 *ptr) { float2 ret; asm volatile ("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float4 __ldcg(const float4 *ptr) { float4 ret; asm volatile ("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double2 __ldcg(const double2 *ptr) { double2 ret; asm volatile ("ld.global.cg.v2.f64 {%0,%1}, [%2];" : "=d"(ret.x), "=d"(ret.y) : "l" (ptr)); return ret; }







static __attribute__((device)) __inline__ long __ldca(const long *ptr) { unsigned long ret; asm volatile ("ld.global.ca.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long)ret; }
static __attribute__((device)) __inline__ unsigned long __ldca(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.ca.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }






static __attribute__((device)) __inline__ char __ldca(const char *ptr) { unsigned int ret; asm volatile ("ld.global.ca.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (char)ret; }
static __attribute__((device)) __inline__ signed char __ldca(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.ca.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (signed char)ret; }
static __attribute__((device)) __inline__ short __ldca(const short *ptr) { unsigned short ret; asm volatile ("ld.global.ca.s16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return (short)ret; }
static __attribute__((device)) __inline__ int __ldca(const int *ptr) { unsigned int ret; asm volatile ("ld.global.ca.s32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (int)ret; }
static __attribute__((device)) __inline__ long long __ldca(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.ca.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long long)ret; }
static __attribute__((device)) __inline__ char2 __ldca(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.ca.v2.s8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
static __attribute__((device)) __inline__ char4 __ldca(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.ca.v4.s8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
static __attribute__((device)) __inline__ short2 __ldca(const short2 *ptr) { short2 ret; asm volatile ("ld.global.ca.v2.s16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ short4 __ldca(const short4 *ptr) { short4 ret; asm volatile ("ld.global.ca.v4.s16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int2 __ldca(const int2 *ptr) { int2 ret; asm volatile ("ld.global.ca.v2.s32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int4 __ldca(const int4 *ptr) { int4 ret; asm volatile ("ld.global.ca.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ longlong2 __ldca(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.ca.v2.s64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ unsigned char __ldca(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.ca.u8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (unsigned char)ret; }
static __attribute__((device)) __inline__ unsigned short __ldca(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.ca.u16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned int __ldca(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned long long __ldca(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.ca.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uchar2 __ldca(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.ca.v2.u8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
static __attribute__((device)) __inline__ uchar4 __ldca(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.ca.v4.u8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
static __attribute__((device)) __inline__ ushort2 __ldca(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.ca.v2.u16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ushort4 __ldca(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.ca.v4.u16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint2 __ldca(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.ca.v2.u32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint4 __ldca(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ulonglong2 __ldca(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.ca.v2.u64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ float __ldca(const float *ptr) { float ret; asm volatile ("ld.global.ca.f32 %0, [%1];" : "=f"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double __ldca(const double *ptr) { double ret; asm volatile ("ld.global.ca.f64 %0, [%1];" : "=d"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float2 __ldca(const float2 *ptr) { float2 ret; asm volatile ("ld.global.ca.v2.f32 {%0,%1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float4 __ldca(const float4 *ptr) { float4 ret; asm volatile ("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double2 __ldca(const double2 *ptr) { double2 ret; asm volatile ("ld.global.ca.v2.f64 {%0,%1}, [%2];" : "=d"(ret.x), "=d"(ret.y) : "l" (ptr)); return ret; }







static __attribute__((device)) __inline__ long __ldcs(const long *ptr) { unsigned long ret; asm volatile ("ld.global.cs.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long)ret; }
static __attribute__((device)) __inline__ unsigned long __ldcs(const unsigned long *ptr) { unsigned long ret; asm volatile ("ld.global.cs.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }






static __attribute__((device)) __inline__ char __ldcs(const char *ptr) { unsigned int ret; asm volatile ("ld.global.cs.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (char)ret; }
static __attribute__((device)) __inline__ signed char __ldcs(const signed char *ptr) { unsigned int ret; asm volatile ("ld.global.cs.s8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (signed char)ret; }
static __attribute__((device)) __inline__ short __ldcs(const short *ptr) { unsigned short ret; asm volatile ("ld.global.cs.s16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return (short)ret; }
static __attribute__((device)) __inline__ int __ldcs(const int *ptr) { unsigned int ret; asm volatile ("ld.global.cs.s32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (int)ret; }
static __attribute__((device)) __inline__ long long __ldcs(const long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cs.s64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return (long long)ret; }
static __attribute__((device)) __inline__ char2 __ldcs(const char2 *ptr) { char2 ret; int2 tmp; asm volatile ("ld.global.cs.v2.s8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
static __attribute__((device)) __inline__ char4 __ldcs(const char4 *ptr) { char4 ret; int4 tmp; asm volatile ("ld.global.cs.v4.s8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
static __attribute__((device)) __inline__ short2 __ldcs(const short2 *ptr) { short2 ret; asm volatile ("ld.global.cs.v2.s16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ short4 __ldcs(const short4 *ptr) { short4 ret; asm volatile ("ld.global.cs.v4.s16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int2 __ldcs(const int2 *ptr) { int2 ret; asm volatile ("ld.global.cs.v2.s32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ int4 __ldcs(const int4 *ptr) { int4 ret; asm volatile ("ld.global.cs.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ longlong2 __ldcs(const longlong2 *ptr) { longlong2 ret; asm volatile ("ld.global.cs.v2.s64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ unsigned char __ldcs(const unsigned char *ptr) { unsigned int ret; asm volatile ("ld.global.cs.u8 %0, [%1];" : "=r"(ret) : "l" (ptr)); return (unsigned char)ret; }
static __attribute__((device)) __inline__ unsigned short __ldcs(const unsigned short *ptr) { unsigned short ret; asm volatile ("ld.global.cs.u16 %0, [%1];" : "=h"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned int __ldcs(const unsigned int *ptr) { unsigned int ret; asm volatile ("ld.global.cs.u32 %0, [%1];" : "=r"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ unsigned long long __ldcs(const unsigned long long *ptr) { unsigned long long ret; asm volatile ("ld.global.cs.u64 %0, [%1];" : "=l"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uchar2 __ldcs(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm volatile ("ld.global.cs.v2.u8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
static __attribute__((device)) __inline__ uchar4 __ldcs(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm volatile ("ld.global.cs.v4.u8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr)); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
static __attribute__((device)) __inline__ ushort2 __ldcs(const ushort2 *ptr) { ushort2 ret; asm volatile ("ld.global.cs.v2.u16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ushort4 __ldcs(const ushort4 *ptr) { ushort4 ret; asm volatile ("ld.global.cs.v4.u16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint2 __ldcs(const uint2 *ptr) { uint2 ret; asm volatile ("ld.global.cs.v2.u32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ uint4 __ldcs(const uint4 *ptr) { uint4 ret; asm volatile ("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ ulonglong2 __ldcs(const ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.cs.v2.u64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr)); return ret; }

static __attribute__((device)) __inline__ float __ldcs(const float *ptr) { float ret; asm volatile ("ld.global.cs.f32 %0, [%1];" : "=f"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double __ldcs(const double *ptr) { double ret; asm volatile ("ld.global.cs.f64 %0, [%1];" : "=d"(ret) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float2 __ldcs(const float2 *ptr) { float2 ret; asm volatile ("ld.global.cs.v2.f32 {%0,%1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ float4 __ldcs(const float4 *ptr) { float4 ret; asm volatile ("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l" (ptr)); return ret; }
static __attribute__((device)) __inline__ double2 __ldcs(const double2 *ptr) { double2 ret; asm volatile ("ld.global.cs.v2.f64 {%0,%1}, [%2];" : "=d"(ret.x), "=d"(ret.y) : "l" (ptr)); return ret; }







static __attribute__((device)) __inline__ long __ldlu(const long *ptr) { unsigned long ret; asm ("ld.global.lu.s64 %0, [%1];" : "=l"(ret) : "l" (ptr) : "memory"); return (long)ret; }
static __attribute__((device)) __inline__ unsigned long __ldlu(const unsigned long *ptr) { unsigned long ret; asm ("ld.global.lu.u64 %0, [%1];" : "=l"(ret) : "l" (ptr) : "memory"); return ret; }






static __attribute__((device)) __inline__ char __ldlu(const char *ptr) { unsigned int ret; asm ("ld.global.lu.s8 %0, [%1];" : "=r"(ret) : "l" (ptr) : "memory"); return (char)ret; }
static __attribute__((device)) __inline__ signed char __ldlu(const signed char *ptr) { unsigned int ret; asm ("ld.global.lu.s8 %0, [%1];" : "=r"(ret) : "l" (ptr) : "memory"); return (signed char)ret; }
static __attribute__((device)) __inline__ short __ldlu(const short *ptr) { unsigned short ret; asm ("ld.global.lu.s16 %0, [%1];" : "=h"(ret) : "l" (ptr) : "memory"); return (short)ret; }
static __attribute__((device)) __inline__ int __ldlu(const int *ptr) { unsigned int ret; asm ("ld.global.lu.s32 %0, [%1];" : "=r"(ret) : "l" (ptr) : "memory"); return (int)ret; }
static __attribute__((device)) __inline__ long long __ldlu(const long long *ptr) { unsigned long long ret; asm ("ld.global.lu.s64 %0, [%1];" : "=l"(ret) : "l" (ptr) : "memory"); return (long long)ret; }
static __attribute__((device)) __inline__ char2 __ldlu(const char2 *ptr) { char2 ret; int2 tmp; asm ("ld.global.lu.v2.s8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr) : "memory"); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
static __attribute__((device)) __inline__ char4 __ldlu(const char4 *ptr) { char4 ret; int4 tmp; asm ("ld.global.lu.v4.s8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr) : "memory"); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
static __attribute__((device)) __inline__ short2 __ldlu(const short2 *ptr) { short2 ret; asm ("ld.global.lu.v2.s16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ short4 __ldlu(const short4 *ptr) { short4 ret; asm ("ld.global.lu.v4.s16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ int2 __ldlu(const int2 *ptr) { int2 ret; asm ("ld.global.lu.v2.s32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ int4 __ldlu(const int4 *ptr) { int4 ret; asm ("ld.global.lu.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ longlong2 __ldlu(const longlong2 *ptr) { longlong2 ret; asm ("ld.global.lu.v2.s64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr) : "memory"); return ret; }

static __attribute__((device)) __inline__ unsigned char __ldlu(const unsigned char *ptr) { unsigned int ret; asm ("ld.global.lu.u8 %0, [%1];" : "=r"(ret) : "l" (ptr) : "memory"); return (unsigned char)ret; }
static __attribute__((device)) __inline__ unsigned short __ldlu(const unsigned short *ptr) { unsigned short ret; asm ("ld.global.lu.u16 %0, [%1];" : "=h"(ret) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ unsigned int __ldlu(const unsigned int *ptr) { unsigned int ret; asm ("ld.global.lu.u32 %0, [%1];" : "=r"(ret) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ unsigned long long __ldlu(const unsigned long long *ptr) { unsigned long long ret; asm ("ld.global.lu.u64 %0, [%1];" : "=l"(ret) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ uchar2 __ldlu(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm ("ld.global.lu.v2.u8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr) : "memory"); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
static __attribute__((device)) __inline__ uchar4 __ldlu(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm ("ld.global.lu.v4.u8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr) : "memory"); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
static __attribute__((device)) __inline__ ushort2 __ldlu(const ushort2 *ptr) { ushort2 ret; asm ("ld.global.lu.v2.u16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ ushort4 __ldlu(const ushort4 *ptr) { ushort4 ret; asm ("ld.global.lu.v4.u16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ uint2 __ldlu(const uint2 *ptr) { uint2 ret; asm ("ld.global.lu.v2.u32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ uint4 __ldlu(const uint4 *ptr) { uint4 ret; asm ("ld.global.lu.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ ulonglong2 __ldlu(const ulonglong2 *ptr) { ulonglong2 ret; asm ("ld.global.lu.v2.u64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr) : "memory"); return ret; }

static __attribute__((device)) __inline__ float __ldlu(const float *ptr) { float ret; asm ("ld.global.lu.f32 %0, [%1];" : "=f"(ret) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ double __ldlu(const double *ptr) { double ret; asm ("ld.global.lu.f64 %0, [%1];" : "=d"(ret) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ float2 __ldlu(const float2 *ptr) { float2 ret; asm ("ld.global.lu.v2.f32 {%0,%1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ float4 __ldlu(const float4 *ptr) { float4 ret; asm ("ld.global.lu.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ double2 __ldlu(const double2 *ptr) { double2 ret; asm ("ld.global.lu.v2.f64 {%0,%1}, [%2];" : "=d"(ret.x), "=d"(ret.y) : "l" (ptr) : "memory"); return ret; }







static __attribute__((device)) __inline__ long __ldcv(const long *ptr) { unsigned long ret; asm ("ld.global.cv.s64 %0, [%1];" : "=l"(ret) : "l" (ptr) : "memory"); return (long)ret; }
static __attribute__((device)) __inline__ unsigned long __ldcv(const unsigned long *ptr) { unsigned long ret; asm ("ld.global.cv.u64 %0, [%1];" : "=l"(ret) : "l" (ptr) : "memory"); return ret; }






static __attribute__((device)) __inline__ char __ldcv(const char *ptr) { unsigned int ret; asm ("ld.global.cv.s8 %0, [%1];" : "=r"(ret) : "l" (ptr) : "memory"); return (char)ret; }
static __attribute__((device)) __inline__ signed char __ldcv(const signed char *ptr) { unsigned int ret; asm ("ld.global.cv.s8 %0, [%1];" : "=r"(ret) : "l" (ptr) : "memory"); return (signed char)ret; }
static __attribute__((device)) __inline__ short __ldcv(const short *ptr) { unsigned short ret; asm ("ld.global.cv.s16 %0, [%1];" : "=h"(ret) : "l" (ptr) : "memory"); return (short)ret; }
static __attribute__((device)) __inline__ int __ldcv(const int *ptr) { unsigned int ret; asm ("ld.global.cv.s32 %0, [%1];" : "=r"(ret) : "l" (ptr) : "memory"); return (int)ret; }
static __attribute__((device)) __inline__ long long __ldcv(const long long *ptr) { unsigned long long ret; asm ("ld.global.cv.s64 %0, [%1];" : "=l"(ret) : "l" (ptr) : "memory"); return (long long)ret; }
static __attribute__((device)) __inline__ char2 __ldcv(const char2 *ptr) { char2 ret; int2 tmp; asm ("ld.global.cv.v2.s8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr) : "memory"); ret.x = (char)tmp.x; ret.y = (char)tmp.y; return ret; }
static __attribute__((device)) __inline__ char4 __ldcv(const char4 *ptr) { char4 ret; int4 tmp; asm ("ld.global.cv.v4.s8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr) : "memory"); ret.x = (char)tmp.x; ret.y = (char)tmp.y; ret.z = (char)tmp.z; ret.w = (char)tmp.w; return ret; }
static __attribute__((device)) __inline__ short2 __ldcv(const short2 *ptr) { short2 ret; asm ("ld.global.cv.v2.s16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ short4 __ldcv(const short4 *ptr) { short4 ret; asm ("ld.global.cv.v4.s16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ int2 __ldcv(const int2 *ptr) { int2 ret; asm ("ld.global.cv.v2.s32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ int4 __ldcv(const int4 *ptr) { int4 ret; asm ("ld.global.cv.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ longlong2 __ldcv(const longlong2 *ptr) { longlong2 ret; asm ("ld.global.cv.v2.s64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr) : "memory"); return ret; }

static __attribute__((device)) __inline__ unsigned char __ldcv(const unsigned char *ptr) { unsigned int ret; asm ("ld.global.cv.u8 %0, [%1];" : "=r"(ret) : "l" (ptr) : "memory"); return (unsigned char)ret; }
static __attribute__((device)) __inline__ unsigned short __ldcv(const unsigned short *ptr) { unsigned short ret; asm ("ld.global.cv.u16 %0, [%1];" : "=h"(ret) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ unsigned int __ldcv(const unsigned int *ptr) { unsigned int ret; asm ("ld.global.cv.u32 %0, [%1];" : "=r"(ret) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ unsigned long long __ldcv(const unsigned long long *ptr) { unsigned long long ret; asm ("ld.global.cv.u64 %0, [%1];" : "=l"(ret) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ uchar2 __ldcv(const uchar2 *ptr) { uchar2 ret; uint2 tmp; asm ("ld.global.cv.v2.u8 {%0,%1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l" (ptr) : "memory"); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; return ret; }
static __attribute__((device)) __inline__ uchar4 __ldcv(const uchar4 *ptr) { uchar4 ret; uint4 tmp; asm ("ld.global.cv.v4.u8 {%0,%1,%2,%3}, [%4];" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l" (ptr) : "memory"); ret.x = (unsigned char)tmp.x; ret.y = (unsigned char)tmp.y; ret.z = (unsigned char)tmp.z; ret.w = (unsigned char)tmp.w; return ret; }
static __attribute__((device)) __inline__ ushort2 __ldcv(const ushort2 *ptr) { ushort2 ret; asm ("ld.global.cv.v2.u16 {%0,%1}, [%2];" : "=h"(ret.x), "=h"(ret.y) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ ushort4 __ldcv(const ushort4 *ptr) { ushort4 ret; asm ("ld.global.cv.v4.u16 {%0,%1,%2,%3}, [%4];" : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ uint2 __ldcv(const uint2 *ptr) { uint2 ret; asm ("ld.global.cv.v2.u32 {%0,%1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ uint4 __ldcv(const uint4 *ptr) { uint4 ret; asm ("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ ulonglong2 __ldcv(const ulonglong2 *ptr) { ulonglong2 ret; asm ("ld.global.cv.v2.u64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : "l" (ptr) : "memory"); return ret; }

static __attribute__((device)) __inline__ float __ldcv(const float *ptr) { float ret; asm ("ld.global.cv.f32 %0, [%1];" : "=f"(ret) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ double __ldcv(const double *ptr) { double ret; asm ("ld.global.cv.f64 %0, [%1];" : "=d"(ret) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ float2 __ldcv(const float2 *ptr) { float2 ret; asm ("ld.global.cv.v2.f32 {%0,%1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ float4 __ldcv(const float4 *ptr) { float4 ret; asm ("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l" (ptr) : "memory"); return ret; }
static __attribute__((device)) __inline__ double2 __ldcv(const double2 *ptr) { double2 ret; asm ("ld.global.cv.v2.f64 {%0,%1}, [%2];" : "=d"(ret.x), "=d"(ret.y) : "l" (ptr) : "memory"); return ret; }







static __attribute__((device)) __inline__ void __stwb(long *ptr, long value) { asm ("st.global.wb.s64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(unsigned long *ptr, unsigned long value) { asm ("st.global.wb.u64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }






static __attribute__((device)) __inline__ void __stwb(char *ptr, char value) { asm ("st.global.wb.s8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(signed char *ptr, signed char value) { asm ("st.global.wb.s8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(short *ptr, short value) { asm ("st.global.wb.s16 [%0], %1;" :: "l" (ptr), "h"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(int *ptr, int value) { asm ("st.global.wb.s32 [%0], %1;" :: "l" (ptr), "r"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(long long *ptr, long long value) { asm ("st.global.wb.s64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(char2 *ptr, char2 value) { const int x = value.x, y = value.y; asm ("st.global.wb.v2.s8 [%0], {%1,%2};" :: "l" (ptr), "r"(x), "r"(y) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(char4 *ptr, char4 value) { const int x = value.x, y = value.y, z = value.z, w = value.w; asm ("st.global.wb.v4.s8 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(short2 *ptr, short2 value) { asm ("st.global.wb.v2.s16 [%0], {%1,%2};" :: "l" (ptr), "h"(value.x), "h"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(short4 *ptr, short4 value) { asm ("st.global.wb.v4.s16 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "h"(value.x), "h"(value.y), "h"(value.z), "h"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(int2 *ptr, int2 value) { asm ("st.global.wb.v2.s32 [%0], {%1,%2};" :: "l" (ptr), "r"(value.x), "r"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(int4 *ptr, int4 value) { asm ("st.global.wb.v4.s32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(longlong2 *ptr, longlong2 value) { asm ("st.global.wb.v2.s64 [%0], {%1,%2};" :: "l" (ptr), "l"(value.x), "l"(value.y) : "memory"); }

static __attribute__((device)) __inline__ void __stwb(unsigned char *ptr, unsigned char value) { asm ("st.global.wb.u8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(unsigned short *ptr, unsigned short value) { asm ("st.global.wb.u16 [%0], %1;" :: "l" (ptr), "h"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(unsigned int *ptr, unsigned int value) { asm ("st.global.wb.u32 [%0], %1;" :: "l" (ptr), "r"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(unsigned long long *ptr, unsigned long long value) { asm ("st.global.wb.u64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(uchar2 *ptr, uchar2 value) { const int x = value.x, y = value.y; asm ("st.global.wb.v2.u8 [%0], {%1,%2};" :: "l" (ptr), "r"(x), "r"(y) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(uchar4 *ptr, uchar4 value) { const int x = value.x, y = value.y, z = value.z, w = value.w; asm ("st.global.wb.v4.u8 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(ushort2 *ptr, ushort2 value) { asm ("st.global.wb.v2.u16 [%0], {%1,%2};" :: "l" (ptr), "h"(value.x), "h"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(ushort4 *ptr, ushort4 value) { asm ("st.global.wb.v4.u16 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "h"(value.x), "h"(value.y), "h"(value.z), "h"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(uint2 *ptr, uint2 value) { asm ("st.global.wb.v2.u32 [%0], {%1,%2};" :: "l" (ptr), "r"(value.x), "r"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(uint4 *ptr, uint4 value) { asm ("st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(ulonglong2 *ptr, ulonglong2 value) { asm ("st.global.wb.v2.u64 [%0], {%1,%2};" :: "l" (ptr), "l"(value.x), "l"(value.y) : "memory"); }

static __attribute__((device)) __inline__ void __stwb(float *ptr, float value) { asm ("st.global.wb.f32 [%0], %1;" :: "l" (ptr), "f"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(double *ptr, double value) { asm ("st.global.wb.f64 [%0], %1;" :: "l" (ptr), "d"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(float2 *ptr, float2 value) { asm ("st.global.wb.v2.f32 [%0], {%1,%2};" :: "l" (ptr), "f"(value.x), "f"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(float4 *ptr, float4 value) { asm ("st.global.wb.v4.f32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stwb(double2 *ptr, double2 value) { asm ("st.global.wb.v2.f64 [%0], {%1,%2};" :: "l" (ptr), "d"(value.x), "d"(value.y) : "memory"); }







static __attribute__((device)) __inline__ void __stcg(long *ptr, long value) { asm ("st.global.cg.s64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(unsigned long *ptr, unsigned long value) { asm ("st.global.cg.u64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }






static __attribute__((device)) __inline__ void __stcg(char *ptr, char value) { asm ("st.global.cg.s8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(signed char *ptr, signed char value) { asm ("st.global.cg.s8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(short *ptr, short value) { asm ("st.global.cg.s16 [%0], %1;" :: "l" (ptr), "h"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(int *ptr, int value) { asm ("st.global.cg.s32 [%0], %1;" :: "l" (ptr), "r"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(long long *ptr, long long value) { asm ("st.global.cg.s64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(char2 *ptr, char2 value) { const int x = value.x, y = value.y; asm ("st.global.cg.v2.s8 [%0], {%1,%2};" :: "l" (ptr), "r"(x), "r"(y) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(char4 *ptr, char4 value) { const int x = value.x, y = value.y, z = value.z, w = value.w; asm ("st.global.cg.v4.s8 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(short2 *ptr, short2 value) { asm ("st.global.cg.v2.s16 [%0], {%1,%2};" :: "l" (ptr), "h"(value.x), "h"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(short4 *ptr, short4 value) { asm ("st.global.cg.v4.s16 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "h"(value.x), "h"(value.y), "h"(value.z), "h"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(int2 *ptr, int2 value) { asm ("st.global.cg.v2.s32 [%0], {%1,%2};" :: "l" (ptr), "r"(value.x), "r"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(int4 *ptr, int4 value) { asm ("st.global.cg.v4.s32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(longlong2 *ptr, longlong2 value) { asm ("st.global.cg.v2.s64 [%0], {%1,%2};" :: "l" (ptr), "l"(value.x), "l"(value.y) : "memory"); }

static __attribute__((device)) __inline__ void __stcg(unsigned char *ptr, unsigned char value) { asm ("st.global.cg.u8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(unsigned short *ptr, unsigned short value) { asm ("st.global.cg.u16 [%0], %1;" :: "l" (ptr), "h"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(unsigned int *ptr, unsigned int value) { asm ("st.global.cg.u32 [%0], %1;" :: "l" (ptr), "r"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(unsigned long long *ptr, unsigned long long value) { asm ("st.global.cg.u64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(uchar2 *ptr, uchar2 value) { const int x = value.x, y = value.y; asm ("st.global.cg.v2.u8 [%0], {%1,%2};" :: "l" (ptr), "r"(x), "r"(y) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(uchar4 *ptr, uchar4 value) { const int x = value.x, y = value.y, z = value.z, w = value.w; asm ("st.global.cg.v4.u8 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(ushort2 *ptr, ushort2 value) { asm ("st.global.cg.v2.u16 [%0], {%1,%2};" :: "l" (ptr), "h"(value.x), "h"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(ushort4 *ptr, ushort4 value) { asm ("st.global.cg.v4.u16 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "h"(value.x), "h"(value.y), "h"(value.z), "h"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(uint2 *ptr, uint2 value) { asm ("st.global.cg.v2.u32 [%0], {%1,%2};" :: "l" (ptr), "r"(value.x), "r"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(uint4 *ptr, uint4 value) { asm ("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(ulonglong2 *ptr, ulonglong2 value) { asm ("st.global.cg.v2.u64 [%0], {%1,%2};" :: "l" (ptr), "l"(value.x), "l"(value.y) : "memory"); }

static __attribute__((device)) __inline__ void __stcg(float *ptr, float value) { asm ("st.global.cg.f32 [%0], %1;" :: "l" (ptr), "f"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(double *ptr, double value) { asm ("st.global.cg.f64 [%0], %1;" :: "l" (ptr), "d"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(float2 *ptr, float2 value) { asm ("st.global.cg.v2.f32 [%0], {%1,%2};" :: "l" (ptr), "f"(value.x), "f"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(float4 *ptr, float4 value) { asm ("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stcg(double2 *ptr, double2 value) { asm ("st.global.cg.v2.f64 [%0], {%1,%2};" :: "l" (ptr), "d"(value.x), "d"(value.y) : "memory"); }







static __attribute__((device)) __inline__ void __stcs(long *ptr, long value) { asm ("st.global.cs.s64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(unsigned long *ptr, unsigned long value) { asm ("st.global.cs.u64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }






static __attribute__((device)) __inline__ void __stcs(char *ptr, char value) { asm ("st.global.cs.s8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(signed char *ptr, signed char value) { asm ("st.global.cs.s8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(short *ptr, short value) { asm ("st.global.cs.s16 [%0], %1;" :: "l" (ptr), "h"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(int *ptr, int value) { asm ("st.global.cs.s32 [%0], %1;" :: "l" (ptr), "r"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(long long *ptr, long long value) { asm ("st.global.cs.s64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(char2 *ptr, char2 value) { const int x = value.x, y = value.y; asm ("st.global.cs.v2.s8 [%0], {%1,%2};" :: "l" (ptr), "r"(x), "r"(y) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(char4 *ptr, char4 value) { const int x = value.x, y = value.y, z = value.z, w = value.w; asm ("st.global.cs.v4.s8 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(short2 *ptr, short2 value) { asm ("st.global.cs.v2.s16 [%0], {%1,%2};" :: "l" (ptr), "h"(value.x), "h"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(short4 *ptr, short4 value) { asm ("st.global.cs.v4.s16 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "h"(value.x), "h"(value.y), "h"(value.z), "h"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(int2 *ptr, int2 value) { asm ("st.global.cs.v2.s32 [%0], {%1,%2};" :: "l" (ptr), "r"(value.x), "r"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(int4 *ptr, int4 value) { asm ("st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(longlong2 *ptr, longlong2 value) { asm ("st.global.cs.v2.s64 [%0], {%1,%2};" :: "l" (ptr), "l"(value.x), "l"(value.y) : "memory"); }

static __attribute__((device)) __inline__ void __stcs(unsigned char *ptr, unsigned char value) { asm ("st.global.cs.u8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(unsigned short *ptr, unsigned short value) { asm ("st.global.cs.u16 [%0], %1;" :: "l" (ptr), "h"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(unsigned int *ptr, unsigned int value) { asm ("st.global.cs.u32 [%0], %1;" :: "l" (ptr), "r"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(unsigned long long *ptr, unsigned long long value) { asm ("st.global.cs.u64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(uchar2 *ptr, uchar2 value) { const int x = value.x, y = value.y; asm ("st.global.cs.v2.u8 [%0], {%1,%2};" :: "l" (ptr), "r"(x), "r"(y) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(uchar4 *ptr, uchar4 value) { const int x = value.x, y = value.y, z = value.z, w = value.w; asm ("st.global.cs.v4.u8 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(ushort2 *ptr, ushort2 value) { asm ("st.global.cs.v2.u16 [%0], {%1,%2};" :: "l" (ptr), "h"(value.x), "h"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(ushort4 *ptr, ushort4 value) { asm ("st.global.cs.v4.u16 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "h"(value.x), "h"(value.y), "h"(value.z), "h"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(uint2 *ptr, uint2 value) { asm ("st.global.cs.v2.u32 [%0], {%1,%2};" :: "l" (ptr), "r"(value.x), "r"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(uint4 *ptr, uint4 value) { asm ("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(ulonglong2 *ptr, ulonglong2 value) { asm ("st.global.cs.v2.u64 [%0], {%1,%2};" :: "l" (ptr), "l"(value.x), "l"(value.y) : "memory"); }

static __attribute__((device)) __inline__ void __stcs(float *ptr, float value) { asm ("st.global.cs.f32 [%0], %1;" :: "l" (ptr), "f"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(double *ptr, double value) { asm ("st.global.cs.f64 [%0], %1;" :: "l" (ptr), "d"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(float2 *ptr, float2 value) { asm ("st.global.cs.v2.f32 [%0], {%1,%2};" :: "l" (ptr), "f"(value.x), "f"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(float4 *ptr, float4 value) { asm ("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stcs(double2 *ptr, double2 value) { asm ("st.global.cs.v2.f64 [%0], {%1,%2};" :: "l" (ptr), "d"(value.x), "d"(value.y) : "memory"); }







static __attribute__((device)) __inline__ void __stwt(long *ptr, long value) { asm ("st.global.wt.s64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(unsigned long *ptr, unsigned long value) { asm ("st.global.wt.u64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }






static __attribute__((device)) __inline__ void __stwt(char *ptr, char value) { asm ("st.global.wt.s8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(signed char *ptr, signed char value) { asm ("st.global.wt.s8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(short *ptr, short value) { asm ("st.global.wt.s16 [%0], %1;" :: "l" (ptr), "h"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(int *ptr, int value) { asm ("st.global.wt.s32 [%0], %1;" :: "l" (ptr), "r"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(long long *ptr, long long value) { asm ("st.global.wt.s64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(char2 *ptr, char2 value) { const int x = value.x, y = value.y; asm ("st.global.wt.v2.s8 [%0], {%1,%2};" :: "l" (ptr), "r"(x), "r"(y) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(char4 *ptr, char4 value) { const int x = value.x, y = value.y, z = value.z, w = value.w; asm ("st.global.wt.v4.s8 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(short2 *ptr, short2 value) { asm ("st.global.wt.v2.s16 [%0], {%1,%2};" :: "l" (ptr), "h"(value.x), "h"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(short4 *ptr, short4 value) { asm ("st.global.wt.v4.s16 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "h"(value.x), "h"(value.y), "h"(value.z), "h"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(int2 *ptr, int2 value) { asm ("st.global.wt.v2.s32 [%0], {%1,%2};" :: "l" (ptr), "r"(value.x), "r"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(int4 *ptr, int4 value) { asm ("st.global.wt.v4.s32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(longlong2 *ptr, longlong2 value) { asm ("st.global.wt.v2.s64 [%0], {%1,%2};" :: "l" (ptr), "l"(value.x), "l"(value.y) : "memory"); }

static __attribute__((device)) __inline__ void __stwt(unsigned char *ptr, unsigned char value) { asm ("st.global.wt.u8 [%0], %1;" :: "l" (ptr), "r"((int)value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(unsigned short *ptr, unsigned short value) { asm ("st.global.wt.u16 [%0], %1;" :: "l" (ptr), "h"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(unsigned int *ptr, unsigned int value) { asm ("st.global.wt.u32 [%0], %1;" :: "l" (ptr), "r"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(unsigned long long *ptr, unsigned long long value) { asm ("st.global.wt.u64 [%0], %1;" :: "l" (ptr), "l"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(uchar2 *ptr, uchar2 value) { const int x = value.x, y = value.y; asm ("st.global.wt.v2.u8 [%0], {%1,%2};" :: "l" (ptr), "r"(x), "r"(y) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(uchar4 *ptr, uchar4 value) { const int x = value.x, y = value.y, z = value.z, w = value.w; asm ("st.global.wt.v4.u8 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(ushort2 *ptr, ushort2 value) { asm ("st.global.wt.v2.u16 [%0], {%1,%2};" :: "l" (ptr), "h"(value.x), "h"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(ushort4 *ptr, ushort4 value) { asm ("st.global.wt.v4.u16 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "h"(value.x), "h"(value.y), "h"(value.z), "h"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(uint2 *ptr, uint2 value) { asm ("st.global.wt.v2.u32 [%0], {%1,%2};" :: "l" (ptr), "r"(value.x), "r"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(uint4 *ptr, uint4 value) { asm ("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(ulonglong2 *ptr, ulonglong2 value) { asm ("st.global.wt.v2.u64 [%0], {%1,%2};" :: "l" (ptr), "l"(value.x), "l"(value.y) : "memory"); }

static __attribute__((device)) __inline__ void __stwt(float *ptr, float value) { asm ("st.global.wt.f32 [%0], %1;" :: "l" (ptr), "f"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(double *ptr, double value) { asm ("st.global.wt.f64 [%0], %1;" :: "l" (ptr), "d"(value) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(float2 *ptr, float2 value) { asm ("st.global.wt.v2.f32 [%0], {%1,%2};" :: "l" (ptr), "f"(value.x), "f"(value.y) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(float4 *ptr, float4 value) { asm ("st.global.wt.v4.f32 [%0], {%1,%2,%3,%4};" :: "l" (ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w) : "memory"); }
static __attribute__((device)) __inline__ void __stwt(double2 *ptr, double2 value) { asm ("st.global.wt.v2.f64 [%0], {%1,%2};" :: "l" (ptr), "d"(value.x), "d"(value.y) : "memory"); }
# 553 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.hpp"
static __attribute__((device)) __inline__ unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}
static __attribute__((device)) __inline__ unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}


static __attribute__((device)) __inline__ unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}
static __attribute__((device)) __inline__ unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned int ret;
    asm volatile ("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(shift));
    return ret;
}
# 508 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h" 2
# 3287 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_35_intrinsics.h" 1
# 111 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_35_intrinsics.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h" 1
# 112 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_35_intrinsics.h" 2
# 3288 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h" 1
# 120 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.hpp" 1
# 121 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h" 2
# 3289 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h" 1
# 123 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.hpp" 1
# 124 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h" 2
# 3290 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h" 1
# 114 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h"
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.hpp" 1
# 115 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/sm_80_rt.h" 2
# 3291 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/surface_functions.h" 1
# 65 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template <typename T> struct __nv_surf_trait { typedef void * cast_type; };

template<> struct __nv_surf_trait<char> { typedef char * cast_type; };
template<> struct __nv_surf_trait<signed char> { typedef signed char * cast_type; };
template<> struct __nv_surf_trait<unsigned char> { typedef unsigned char * cast_type; };
template<> struct __nv_surf_trait<char1> { typedef char1 * cast_type; };
template<> struct __nv_surf_trait<uchar1> { typedef uchar1 * cast_type; };
template<> struct __nv_surf_trait<char2> { typedef char2 * cast_type; };
template<> struct __nv_surf_trait<uchar2> { typedef uchar2 * cast_type; };
template<> struct __nv_surf_trait<char4> { typedef char4 * cast_type; };
template<> struct __nv_surf_trait<uchar4> { typedef uchar4 * cast_type; };
template<> struct __nv_surf_trait<short> { typedef short * cast_type; };
template<> struct __nv_surf_trait<unsigned short> { typedef unsigned short * cast_type; };
template<> struct __nv_surf_trait<short1> { typedef short1 * cast_type; };
template<> struct __nv_surf_trait<ushort1> { typedef ushort1 * cast_type; };
template<> struct __nv_surf_trait<short2> { typedef short2 * cast_type; };
template<> struct __nv_surf_trait<ushort2> { typedef ushort2 * cast_type; };
template<> struct __nv_surf_trait<short4> { typedef short4 * cast_type; };
template<> struct __nv_surf_trait<ushort4> { typedef ushort4 * cast_type; };
template<> struct __nv_surf_trait<int> { typedef int * cast_type; };
template<> struct __nv_surf_trait<unsigned int> { typedef unsigned int * cast_type; };
template<> struct __nv_surf_trait<int1> { typedef int1 * cast_type; };
template<> struct __nv_surf_trait<uint1> { typedef uint1 * cast_type; };
template<> struct __nv_surf_trait<int2> { typedef int2 * cast_type; };
template<> struct __nv_surf_trait<uint2> { typedef uint2 * cast_type; };
template<> struct __nv_surf_trait<int4> { typedef int4 * cast_type; };
template<> struct __nv_surf_trait<uint4> { typedef uint4 * cast_type; };
template<> struct __nv_surf_trait<long long> { typedef long long * cast_type; };
template<> struct __nv_surf_trait<unsigned long long> { typedef unsigned long long * cast_type; };
template<> struct __nv_surf_trait<longlong1> { typedef longlong1 * cast_type; };
template<> struct __nv_surf_trait<ulonglong1> { typedef ulonglong1 * cast_type; };
template<> struct __nv_surf_trait<longlong2> { typedef longlong2 * cast_type; };
template<> struct __nv_surf_trait<ulonglong2> { typedef ulonglong2 * cast_type; };
# 108 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/surface_functions.h"
template<> struct __nv_surf_trait<float> { typedef float * cast_type; };
template<> struct __nv_surf_trait<float1> { typedef float1 * cast_type; };
template<> struct __nv_surf_trait<float2> { typedef float2 * cast_type; };
template<> struct __nv_surf_trait<float4> { typedef float4 * cast_type; };


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1Dread(T *res, surface<void, 0x01> surf, int x, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1Dread_v2", (void *)res, s, surf, x, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surf1Dread(surface<void, 0x01> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surf1Dread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, mode);
  return temp;

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1Dread(T *res, surface<void, 0x01> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surf1Dread<T>(surf, x, mode);

}


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2Dread(T *res, surface<void, 0x02> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2Dread_v2", (void *)res, s, surf, x, y, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surf2Dread(surface<void, 0x02> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surf2Dread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, mode);
  return temp;

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2Dread(T *res, surface<void, 0x02> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surf2Dread<T>(surf, x, y, mode);

}


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf3Dread(T *res, surface<void, 0x03> surf, int x, int y, int z, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf3Dread_v2", (void *)res, s, surf, x, y, z, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surf3Dread(surface<void, 0x03> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surf3Dread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, z, mode);
  return temp;

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf3Dread(T *res, surface<void, 0x03> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surf3Dread<T>(surf, x, y, z, mode);

}



template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1DLayeredread(T *res, surface<void, 0xF1> surf, int x, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1DLayeredread_v2", (void *)res, s, surf, x, layer, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surf1DLayeredread(surface<void, 0xF1> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surf1DLayeredread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, layer, mode);
  return temp;

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1DLayeredread(T *res, surface<void, 0xF1> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surf1DLayeredread<T>(surf, x, layer, mode);

}


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2DLayeredread(T *res, surface<void, 0xF2> surf, int x, int y, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2DLayeredread_v2", (void *)res, s, surf, x, y, layer, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surf2DLayeredread(surface<void, 0xF2> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surf2DLayeredread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, layer, mode);
  return temp;

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2DLayeredread(T *res, surface<void, 0xF2> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surf2DLayeredread<T>(surf, x, y, layer, mode);

}


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapread(T *res, surface<void, 0x0C> surf, int x, int y, int face, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapread_v2", (void *)res, s, surf, x, y, face, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surfCubemapread(surface<void, 0x0C> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;

  __nv_tex_surf_handler("__surfCubemapread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, face, mode);
  return temp;

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapread(T *res, surface<void, 0x0C> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surfCubemapread<T>(surf, x, y, face, mode);

}


template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapLayeredread(T *res, surface<void, 0xFC> surf, int x, int y, int layerFace, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapLayeredread_v2", (void *)res, s, surf, x, y, layerFace, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) T surfCubemapLayeredread(surface<void, 0xFC> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  T temp;
  __nv_tex_surf_handler("__surfCubemapLayeredread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, layerFace, mode);
  return temp;

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapLayeredread(T *res, surface<void, 0xFC> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  *res = surfCubemapLayeredread<T>(surf, x, y, layerFace, mode);

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1Dwrite(T val, surface<void, 0x01> surf, int x, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1Dwrite_v2", (void *)&val, s, surf, x, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1Dwrite(T val, surface<void, 0x01> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1Dwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, mode);

}



template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2Dwrite(T val, surface<void, 0x02> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2Dwrite_v2", (void *)&val, s, surf, x, y, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2Dwrite(T val, surface<void, 0x02> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2Dwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, mode);

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf3Dwrite(T val, surface<void, 0x03> surf, int x, int y, int z, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf3Dwrite_v2", (void *)&val, s, surf, x, y, z,mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf3Dwrite(T val, surface<void, 0x03> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf3Dwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, z, mode);

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1DLayeredwrite(T val, surface<void, 0xF1> surf, int x, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1DLayeredwrite_v2", (void *)&val, s, surf, x, layer,mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf1DLayeredwrite(T val, surface<void, 0xF1> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf1DLayeredwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, layer, mode);

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2DLayeredwrite(T val, surface<void, 0xF2> surf, int x, int y, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2DLayeredwrite_v2", (void *)&val, s, surf, x, y, layer,mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surf2DLayeredwrite(T val, surface<void, 0xF2> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surf2DLayeredwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, layer, mode);

}


template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapwrite(T val, surface<void, 0x0C> surf, int x, int y, int face, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapwrite_v2", (void *)&val, s, surf, x, y, face, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapwrite(T val, surface<void, 0x0C> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, face, mode);

}



template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapLayeredwrite(T val, surface<void, 0xFC> surf, int x, int y, int layerFace, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapLayeredwrite_v2", (void *)&val, s, surf, x, y, layerFace, mode);

}

template<class T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) void surfCubemapLayeredwrite(T val, surface<void, 0xFC> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__surfCubemapLayeredwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, layerFace, mode);

}
# 3292 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h" 1
# 66 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template <typename T>
struct __nv_tex_rmet_ret { };

template<> struct __nv_tex_rmet_ret<char> { typedef char type; };
template<> struct __nv_tex_rmet_ret<signed char> { typedef signed char type; };
template<> struct __nv_tex_rmet_ret<unsigned char> { typedef unsigned char type; };
template<> struct __nv_tex_rmet_ret<char1> { typedef char1 type; };
template<> struct __nv_tex_rmet_ret<uchar1> { typedef uchar1 type; };
template<> struct __nv_tex_rmet_ret<char2> { typedef char2 type; };
template<> struct __nv_tex_rmet_ret<uchar2> { typedef uchar2 type; };
template<> struct __nv_tex_rmet_ret<char4> { typedef char4 type; };
template<> struct __nv_tex_rmet_ret<uchar4> { typedef uchar4 type; };

template<> struct __nv_tex_rmet_ret<short> { typedef short type; };
template<> struct __nv_tex_rmet_ret<unsigned short> { typedef unsigned short type; };
template<> struct __nv_tex_rmet_ret<short1> { typedef short1 type; };
template<> struct __nv_tex_rmet_ret<ushort1> { typedef ushort1 type; };
template<> struct __nv_tex_rmet_ret<short2> { typedef short2 type; };
template<> struct __nv_tex_rmet_ret<ushort2> { typedef ushort2 type; };
template<> struct __nv_tex_rmet_ret<short4> { typedef short4 type; };
template<> struct __nv_tex_rmet_ret<ushort4> { typedef ushort4 type; };

template<> struct __nv_tex_rmet_ret<int> { typedef int type; };
template<> struct __nv_tex_rmet_ret<unsigned int> { typedef unsigned int type; };
template<> struct __nv_tex_rmet_ret<int1> { typedef int1 type; };
template<> struct __nv_tex_rmet_ret<uint1> { typedef uint1 type; };
template<> struct __nv_tex_rmet_ret<int2> { typedef int2 type; };
template<> struct __nv_tex_rmet_ret<uint2> { typedef uint2 type; };
template<> struct __nv_tex_rmet_ret<int4> { typedef int4 type; };
template<> struct __nv_tex_rmet_ret<uint4> { typedef uint4 type; };
# 107 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template<> struct __nv_tex_rmet_ret<float> { typedef float type; };
template<> struct __nv_tex_rmet_ret<float1> { typedef float1 type; };
template<> struct __nv_tex_rmet_ret<float2> { typedef float2 type; };
template<> struct __nv_tex_rmet_ret<float4> { typedef float4 type; };


template <typename T> struct __nv_tex_rmet_cast { typedef T* type; };
# 125 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1Dfetch(texture<T, 0x01, cudaReadModeElementType> t, int x)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1Dfetch_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x);
  return temp;

}

template <typename T>
struct __nv_tex_rmnf_ret { };

template <> struct __nv_tex_rmnf_ret<char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<signed char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<unsigned char> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<short> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<unsigned short> { typedef float type; };
template <> struct __nv_tex_rmnf_ret<char1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<uchar1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<short1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<ushort1> { typedef float1 type; };
template <> struct __nv_tex_rmnf_ret<char2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<uchar2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<short2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<ushort2> { typedef float2 type; };
template <> struct __nv_tex_rmnf_ret<char4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<uchar4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<short4> { typedef float4 type; };
template <> struct __nv_tex_rmnf_ret<ushort4> { typedef float4 type; };

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1Dfetch(texture<T, 0x01, cudaReadModeNormalizedFloat> t, int x)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1Dfetch_rmnf_v2", &type_dummy, &retval, t, x);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1D(texture<T, 0x01, cudaReadModeElementType> t, float x)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1D_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1D(texture<T, 0x01, cudaReadModeNormalizedFloat> t, float x)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1D_rmnf_v2", &type_dummy, &retval, t, x);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2D(texture<T, 0x02, cudaReadModeElementType> t, float x, float y)
{

  typename __nv_tex_rmet_ret<T>::type temp;

  __nv_tex_surf_handler("__tex2D_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x, y);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2D(texture<T, 0x02, cudaReadModeNormalizedFloat> t, float x, float y)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2D_rmnf_v2", &type_dummy, &retval, t, x, y);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1DLayered(texture<T, 0xF1, cudaReadModeElementType> t, float x, int layer)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1DLayered_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x, layer);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1DLayered(texture<T, 0xF1, cudaReadModeNormalizedFloat> t, float x, int layer)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1DLayered_rmnf_v2", &type_dummy, &retval, t, x, layer);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2DLayered(texture<T, 0xF2, cudaReadModeElementType> t, float x, float y, int layer)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex2DLayered_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x, y, layer);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2DLayered(texture<T, 0xF2, cudaReadModeNormalizedFloat> t, float x, float y, int layer)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2DLayered_rmnf_v2", &type_dummy, &retval, t, x, y, layer);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex3D(texture<T, 0x03, cudaReadModeElementType> t, float x, float y, float z)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex3D_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x, y, z);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex3D(texture<T, 0x03, cudaReadModeNormalizedFloat> t, float x, float y, float z)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex3D_rmnf_v2", &type_dummy, &retval, t, x, y, z);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemap(texture<T, 0x0C, cudaReadModeElementType> t, float x, float y, float z)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemap_v2", (typename __nv_tex_rmet_cast<T>::type) &temp, t, x, y, z);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemap(texture<T, 0x0C, cudaReadModeNormalizedFloat> t, float x, float y, float z)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemap_rmnf_v2", &type_dummy, &retval, t, x, y, z);
  return retval;

}


template <typename T>
struct __nv_tex2dgather_ret { };
template <> struct __nv_tex2dgather_ret<char> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<signed char> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char1> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char2> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char3> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<char4> { typedef char4 type; };
template <> struct __nv_tex2dgather_ret<unsigned char> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar1> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar2> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar3> { typedef uchar4 type; };
template <> struct __nv_tex2dgather_ret<uchar4> { typedef uchar4 type; };

template <> struct __nv_tex2dgather_ret<short> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short1> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short2> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short3> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<short4> { typedef short4 type; };
template <> struct __nv_tex2dgather_ret<unsigned short> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort1> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort2> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort3> { typedef ushort4 type; };
template <> struct __nv_tex2dgather_ret<ushort4> { typedef ushort4 type; };

template <> struct __nv_tex2dgather_ret<int> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int1> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int2> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int3> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<int4> { typedef int4 type; };
template <> struct __nv_tex2dgather_ret<unsigned int> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint1> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint2> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint3> { typedef uint4 type; };
template <> struct __nv_tex2dgather_ret<uint4> { typedef uint4 type; };

template <> struct __nv_tex2dgather_ret<float> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float1> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float2> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float3> { typedef float4 type; };
template <> struct __nv_tex2dgather_ret<float4> { typedef float4 type; };

template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) typename __nv_tex2dgather_ret<T>::type tex2Dgather(texture<T, 0x02, cudaReadModeElementType> t, float x, float y, int comp=0)
{

  T type_dummy;
  typename __nv_tex2dgather_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2Dgather_v2", &type_dummy, &retval, t, x, y, comp);
  return retval;

}


template<typename T> struct __nv_tex2dgather_rmnf_ret { };
template<> struct __nv_tex2dgather_rmnf_ret<char> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<signed char> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<unsigned char> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<char4> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<uchar4> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<signed short> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<unsigned short> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort1> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort2> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort3> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<short4> { typedef float4 type; };
template<> struct __nv_tex2dgather_rmnf_ret<ushort4> { typedef float4 type; };

template <typename T>
static __attribute__((device)) __inline__ __attribute__((always_inline)) typename __nv_tex2dgather_rmnf_ret<T>::type tex2Dgather(texture<T, 0x02, cudaReadModeNormalizedFloat> t, float x, float y, int comp = 0)
{

  T type_dummy;
  typename __nv_tex2dgather_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2Dgather_rmnf_v2", &type_dummy, &retval, t, x, y, comp);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1DLod(texture<T, 0x01, cudaReadModeElementType> t, float x, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1DLod_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1DLod(texture<T, 0x01, cudaReadModeNormalizedFloat> t, float x, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1DLod_rmnf_v2", &type_dummy, &retval, t, x, level);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2DLod(texture<T, 0x02, cudaReadModeElementType> t, float x, float y, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex2DLod_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2DLod(texture<T, 0x02, cudaReadModeNormalizedFloat> t, float x, float y, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2DLod_rmnf_v2", &type_dummy, &retval, t, x, y, level);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1DLayeredLod(texture<T, 0xF1, cudaReadModeElementType> t, float x, int layer, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1DLayeredLod_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, layer, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1DLayeredLod(texture<T, 0xF1, cudaReadModeNormalizedFloat> t, float x, int layer, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1DLayeredLod_rmnf_v2", &type_dummy, &retval, t, x, layer, level);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2DLayeredLod(texture<T, 0xF2, cudaReadModeElementType> t, float x, float y, int layer, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex2DLayeredLod_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, layer, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2DLayeredLod(texture<T, 0xF2, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2DLayeredLod_rmnf_v2", &type_dummy, &retval, t, x, y, layer, level);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex3DLod(texture<T, 0x03, cudaReadModeElementType> t, float x, float y, float z, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex3DLod_v2",(typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex3DLod(texture<T, 0x03, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex3DLod_rmnf_v2", &type_dummy, &retval, t, x, y, z, level);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemapLod(texture<T, 0x0C, cudaReadModeElementType> t, float x, float y, float z, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemapLod_v2",(typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemapLod(texture<T, 0x0C, cudaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemapLod_rmnf_v2", &type_dummy, &retval, t, x, y, z, level);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemapLayered(texture<T, 0xFC, cudaReadModeElementType> t, float x, float y, float z, int layer)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemapLayered_v2",(typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, layer);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemapLayered(texture<T, 0xFC, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemapLayered_rmnf_v2", &type_dummy, &retval, t, x, y, z, layer);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemapLayeredLod(texture<T, 0xFC, cudaReadModeElementType> t, float x, float y, float z, int layer, float level)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemapLayeredLod_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, layer, level);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemapLayeredLod(texture<T, 0xFC, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemapLayeredLod_rmnf_v2", &type_dummy, &retval, t, x, y, z, layer, level);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemapGrad(texture<T, 0x0C, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemapGrad_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, &dPdx, &dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemapGrad(texture<T, 0x0C, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemapGrad_rmnf_v2", &type_dummy, &retval, t, x, y, z, &dPdx, &dPdy);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type texCubemapLayeredGrad(texture<T, 0xFC, cudaReadModeElementType> t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__texCubemapLayeredGrad_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, layer, &dPdx, &dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type texCubemapLayeredGrad(texture<T, 0xFC, cudaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__texCubemapLayeredGrad_rmnf_v2", &type_dummy, &retval,t, x, y, z, layer, &dPdx, &dPdy);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1DGrad(texture<T, 0x01, cudaReadModeElementType> t, float x, float dPdx, float dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1DGrad_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, dPdx, dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1DGrad(texture<T, 0x01, cudaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1DGrad_rmnf_v2", &type_dummy, &retval,t, x,dPdx, dPdy);
  return retval;

}



template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2DGrad(texture<T, 0x02, cudaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex2DGrad_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, &dPdx, &dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2DGrad(texture<T, 0x02, cudaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2DGrad_rmnf_v2", &type_dummy, &retval,t, x, y, &dPdx, &dPdy);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex1DLayeredGrad(texture<T, 0xF1, cudaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex1DLayeredGrad_v2",(typename __nv_tex_rmet_cast<T>::type)&temp, t, x, layer, dPdx, dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex1DLayeredGrad(texture<T, 0xF1, cudaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex1DLayeredGrad_rmnf_v2", &type_dummy, &retval,t, x, layer, dPdx, dPdy);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex2DLayeredGrad(texture<T, 0xF2, cudaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex2DLayeredGrad_v2",(typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, layer, &dPdx, &dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex2DLayeredGrad(texture<T, 0xF2, cudaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex2DLayeredGrad_rmnf_v2", &type_dummy, &retval,t, x, y, layer, &dPdx, &dPdy);
  return retval;

}


template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmet_ret<T>::type tex3DGrad(texture<T, 0x03, cudaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  typename __nv_tex_rmet_ret<T>::type temp;
  __nv_tex_surf_handler("__tex3DGrad_v2", (typename __nv_tex_rmet_cast<T>::type)&temp, t, x, y, z, &dPdx, &dPdy);
  return temp;

}

template <typename T>
static __inline__ __attribute__((always_inline)) __attribute__((device)) typename __nv_tex_rmnf_ret<T>::type tex3DGrad(texture<T, 0x03, cudaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  T type_dummy;
  typename __nv_tex_rmnf_ret<T>::type retval;
  __nv_tex_surf_handler("__tex3DGrad_rmnf_v2", &type_dummy, &retval,t, x, y, z, &dPdx, &dPdy);
  return retval;

}
# 3293 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h" 1
# 64 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template <typename T> struct __nv_itex_trait { };
template<> struct __nv_itex_trait<char> { typedef void type; };
template<> struct __nv_itex_trait<signed char> { typedef void type; };
template<> struct __nv_itex_trait<char1> { typedef void type; };
template<> struct __nv_itex_trait<char2> { typedef void type; };
template<> struct __nv_itex_trait<char4> { typedef void type; };
template<> struct __nv_itex_trait<unsigned char> { typedef void type; };
template<> struct __nv_itex_trait<uchar1> { typedef void type; };
template<> struct __nv_itex_trait<uchar2> { typedef void type; };
template<> struct __nv_itex_trait<uchar4> { typedef void type; };
template<> struct __nv_itex_trait<short> { typedef void type; };
template<> struct __nv_itex_trait<short1> { typedef void type; };
template<> struct __nv_itex_trait<short2> { typedef void type; };
template<> struct __nv_itex_trait<short4> { typedef void type; };
template<> struct __nv_itex_trait<unsigned short> { typedef void type; };
template<> struct __nv_itex_trait<ushort1> { typedef void type; };
template<> struct __nv_itex_trait<ushort2> { typedef void type; };
template<> struct __nv_itex_trait<ushort4> { typedef void type; };
template<> struct __nv_itex_trait<int> { typedef void type; };
template<> struct __nv_itex_trait<int1> { typedef void type; };
template<> struct __nv_itex_trait<int2> { typedef void type; };
template<> struct __nv_itex_trait<int4> { typedef void type; };
template<> struct __nv_itex_trait<unsigned int> { typedef void type; };
template<> struct __nv_itex_trait<uint1> { typedef void type; };
template<> struct __nv_itex_trait<uint2> { typedef void type; };
template<> struct __nv_itex_trait<uint4> { typedef void type; };
# 100 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template<> struct __nv_itex_trait<float> { typedef void type; };
template<> struct __nv_itex_trait<float1> { typedef void type; };
template<> struct __nv_itex_trait<float2> { typedef void type; };
template<> struct __nv_itex_trait<float4> { typedef void type; };



template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1Dfetch(T *ptr, cudaTextureObject_t obj, int x)
{

   __nv_tex_surf_handler("__itex1Dfetch", ptr, obj, x);

}

template <class T>
static __attribute__((device)) T tex1Dfetch(cudaTextureObject_t texObject, int x)
{

  T ret;
  tex1Dfetch(&ret, texObject, x);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1D(T *ptr, cudaTextureObject_t obj, float x)
{

   __nv_tex_surf_handler("__itex1D", ptr, obj, x);

}


template <class T>
static __attribute__((device)) T tex1D(cudaTextureObject_t texObject, float x)
{

  T ret;
  tex1D(&ret, texObject, x);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2D(T *ptr, cudaTextureObject_t obj, float x, float y)
{

   __nv_tex_surf_handler("__itex2D", ptr, obj, x, y);

}

template <class T>
static __attribute__((device)) T tex2D(cudaTextureObject_t texObject, float x, float y)
{

  T ret;
  tex2D(&ret, texObject, x, y);
  return ret;

}
# 188 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z)
{

   __nv_tex_surf_handler("__itex3D", ptr, obj, x, y, z);

}

template <class T>
static __attribute__((device)) T tex3D(cudaTextureObject_t texObject, float x, float y, float z)
{

  T ret;
  tex3D(&ret, texObject, x, y, z);
  return ret;

}
# 230 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1DLayered(T *ptr, cudaTextureObject_t obj, float x, int layer)
{

   __nv_tex_surf_handler("__itex1DLayered", ptr, obj, x, layer);

}

template <class T>
static __attribute__((device)) T tex1DLayered(cudaTextureObject_t texObject, float x, int layer)
{

  T ret;
  tex1DLayered(&ret, texObject, x, layer);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer)
{

  __nv_tex_surf_handler("__itex2DLayered", ptr, obj, x, y, layer);

}

template <class T>
static __attribute__((device)) T tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer)
{

  T ret;
  tex2DLayered(&ret, texObject, x, y, layer);
  return ret;

}
# 289 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemap(T *ptr, cudaTextureObject_t obj, float x, float y, float z)
{

  __nv_tex_surf_handler("__itexCubemap", ptr, obj, x, y, z);

}


template <class T>
static __attribute__((device)) T texCubemap(cudaTextureObject_t texObject, float x, float y, float z)
{

  T ret;
  texCubemap(&ret, texObject, x, y, z);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemapLayered(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer)
{

  __nv_tex_surf_handler("__itexCubemapLayered", ptr, obj, x, y, z, layer);

}

template <class T>
static __attribute__((device)) T texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer)
{

  T ret;
  texCubemapLayered(&ret, texObject, x, y, z, layer);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, int comp = 0)
{

  __nv_tex_surf_handler("__itex2Dgather", ptr, obj, x, y, comp);

}

template <class T>
static __attribute__((device)) T tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0)
{

  T ret;
  tex2Dgather(&ret, to, x, y, comp);
  return ret;

}
# 368 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1DLod(T *ptr, cudaTextureObject_t obj, float x, float level)
{

  __nv_tex_surf_handler("__itex1DLod", ptr, obj, x, level);

}

template <class T>
static __attribute__((device)) T tex1DLod(cudaTextureObject_t texObject, float x, float level)
{

  T ret;
  tex1DLod(&ret, texObject, x, level);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level)
{

  __nv_tex_surf_handler("__itex2DLod", ptr, obj, x, y, level);

}

template <class T>
static __attribute__((device)) T tex2DLod(cudaTextureObject_t texObject, float x, float y, float level)
{

  T ret;
  tex2DLod(&ret, texObject, x, y, level);
  return ret;

}
# 430 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level)
{

  __nv_tex_surf_handler("__itex3DLod", ptr, obj, x, y, z, level);

}

template <class T>
static __attribute__((device)) T tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level)
{

  T ret;
  tex3DLod(&ret, texObject, x, y, z, level);
  return ret;

}
# 472 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, int layer, float level)
{

  __nv_tex_surf_handler("__itex1DLayeredLod", ptr, obj, x, layer, level);

}

template <class T>
static __attribute__((device)) T tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level)
{

  T ret;
  tex1DLayeredLod(&ret, texObject, x, layer, level);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level)
{

  __nv_tex_surf_handler("__itex2DLayeredLod", ptr, obj, x, y, layer, level);

}

template <class T>
static __attribute__((device)) T tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level)
{

  T ret;
  tex2DLayeredLod(&ret, texObject, x, y, layer, level);
  return ret;

}
# 531 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemapLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level)
{

  __nv_tex_surf_handler("__itexCubemapLod", ptr, obj, x, y, z, level);

}

template <class T>
static __attribute__((device)) T texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level)
{

  T ret;
  texCubemapLod(&ret, texObject, x, y, z, level);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemapGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  __nv_tex_surf_handler("__itexCubemapGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy);

}

template <class T>
static __attribute__((device)) T texCubemapGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  T ret;
  texCubemapGrad(&ret, texObject, x, y, z, dPdx, dPdy);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemapLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float level)
{

  __nv_tex_surf_handler("__itexCubemapLayeredLod", ptr, obj, x, y, z, layer, level);

}

template <class T>
static __attribute__((device)) T texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{

  T ret;
  texCubemapLayeredLod(&ret, texObject, x, y, z, layer, level);
  return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1DGrad(T *ptr, cudaTextureObject_t obj, float x, float dPdx, float dPdy)
{

  __nv_tex_surf_handler("__itex1DGrad", ptr, obj, x, dPdx, dPdy);

}

template <class T>
static __attribute__((device)) T tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy)
{

  T ret;
  tex1DGrad(&ret, texObject, x, dPdx, dPdy);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy)
{

  __nv_tex_surf_handler("__itex2DGrad_v2", ptr, obj, x, y, &dPdx, &dPdy);


}

template <class T>
static __attribute__((device)) T tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{

  T ret;
  tex2DGrad(&ret, texObject, x, y, dPdx, dPdy);
  return ret;

}
# 648 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  __nv_tex_surf_handler("__itex3DGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy);

}

template <class T>
static __attribute__((device)) T tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{

  T ret;
  tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy);
  return ret;

}
# 690 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex1DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy)
{

  __nv_tex_surf_handler("__itex1DLayeredGrad", ptr, obj, x, layer, dPdx, dPdy);

}

template <class T>
static __attribute__((device)) T tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{

  T ret;
  tex1DLayeredGrad(&ret, texObject, x, layer, dPdx, dPdy);
  return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type tex2DLayeredGrad(T * ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy)
{

  __nv_tex_surf_handler("__itex2DLayeredGrad_v2", ptr, obj, x, y, layer, &dPdx, &dPdy);

}

template <class T>
static __attribute__((device)) T tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{

  T ret;
  tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy);
  return ret;

}
# 750 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template <typename T>
static __attribute__((device)) typename __nv_itex_trait<T>::type texCubemapLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{

  __nv_tex_surf_handler("__itexCubemapLayeredGrad_v2", ptr, obj, x, y, z, layer, &dPdx, &dPdy);

}

template <class T>
static __attribute__((device)) T texCubemapLayeredGrad(cudaTextureObject_t texObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{

  T ret;
  texCubemapLayeredGrad(&ret, texObject, x, y, z, layer, dPdx, dPdy);
  return ret;

}
# 3294 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h" 1
# 59 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template<typename T> struct __nv_isurf_trait { };
template<> struct __nv_isurf_trait<char> { typedef void type; };
template<> struct __nv_isurf_trait<signed char> { typedef void type; };
template<> struct __nv_isurf_trait<char1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned char> { typedef void type; };
template<> struct __nv_isurf_trait<uchar1> { typedef void type; };
template<> struct __nv_isurf_trait<short> { typedef void type; };
template<> struct __nv_isurf_trait<short1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned short> { typedef void type; };
template<> struct __nv_isurf_trait<ushort1> { typedef void type; };
template<> struct __nv_isurf_trait<int> { typedef void type; };
template<> struct __nv_isurf_trait<int1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned int> { typedef void type; };
template<> struct __nv_isurf_trait<uint1> { typedef void type; };
template<> struct __nv_isurf_trait<long long> { typedef void type; };
template<> struct __nv_isurf_trait<longlong1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned long long> { typedef void type; };
template<> struct __nv_isurf_trait<ulonglong1> { typedef void type; };
template<> struct __nv_isurf_trait<float> { typedef void type; };
template<> struct __nv_isurf_trait<float1> { typedef void type; };

template<> struct __nv_isurf_trait<char2> { typedef void type; };
template<> struct __nv_isurf_trait<uchar2> { typedef void type; };
template<> struct __nv_isurf_trait<short2> { typedef void type; };
template<> struct __nv_isurf_trait<ushort2> { typedef void type; };
template<> struct __nv_isurf_trait<int2> { typedef void type; };
template<> struct __nv_isurf_trait<uint2> { typedef void type; };
template<> struct __nv_isurf_trait<longlong2> { typedef void type; };
template<> struct __nv_isurf_trait<ulonglong2> { typedef void type; };
template<> struct __nv_isurf_trait<float2> { typedef void type; };

template<> struct __nv_isurf_trait<char4> { typedef void type; };
template<> struct __nv_isurf_trait<uchar4> { typedef void type; };
template<> struct __nv_isurf_trait<short4> { typedef void type; };
template<> struct __nv_isurf_trait<ushort4> { typedef void type; };
template<> struct __nv_isurf_trait<int4> { typedef void type; };
template<> struct __nv_isurf_trait<uint4> { typedef void type; };
template<> struct __nv_isurf_trait<float4> { typedef void type; };


template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf1Dread(T *ptr, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf1Dread", ptr, obj, x, mode);

}

template <class T>
static __attribute__((device)) T surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surf1Dread(&ret, surfObject, x, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf2Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf2Dread", ptr, obj, x, y, mode);

}

template <class T>
static __attribute__((device)) T surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surf2Dread(&ret, surfObject, x, y, boundaryMode);
   return ret;

}


template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf3Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf3Dread", ptr, obj, x, y, z, mode);

}

template <class T>
static __attribute__((device)) T surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surf3Dread(&ret, surfObject, x, y, z, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf1DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf1DLayeredread", ptr, obj, x, layer, mode);

}

template <class T>
static __attribute__((device)) T surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surf1DLayeredread(&ret, surfObject, x, layer, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf2DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf2DLayeredread", ptr, obj, x, y, layer, mode);

}

template <class T>
static __attribute__((device)) T surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surf2DLayeredread(&ret, surfObject, x, y, layer, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surfCubemapread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurfCubemapread", ptr, obj, x, y, face, mode);

}

template <class T>
static __attribute__((device)) T surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surfCubemapread(&ret, surfObject, x, y, face, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surfCubemapLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurfCubemapLayeredread", ptr, obj, x, y, layerface, mode);

}

template <class T>
static __attribute__((device)) T surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{

   T ret;
   surfCubemapLayeredread(&ret, surfObject, x, y, layerface, boundaryMode);
   return ret;

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf1Dwrite(T val, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, obj, x, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, obj, x, y, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, obj, x, y, z, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf1DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, obj, x, layer, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, obj, x, y, layer, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surfCubemapwrite(T val, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, obj, x, y, face, mode);

}

template <typename T>
static __attribute__((device)) typename __nv_isurf_trait<T>::type surfCubemapLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{

  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, obj, x, y, layerface, mode);

}
# 3295 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/crt/device_functions.h" 2


extern "C" __attribute__((host)) __attribute__((device)) unsigned __cudaPushCallConfiguration(dim3 gridDim,
                                      dim3 blockDim,
                                      size_t sharedMem = 0,
                                      struct CUstream_st *stream = 0);
# 119 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2
# 1 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_launch_parameters.h" 1
# 68 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/device_launch_parameters.h"
extern "C" {


uint3 __attribute__((device_builtin)) extern const threadIdx;
uint3 __attribute__((device_builtin)) extern const blockIdx;
dim3 __attribute__((device_builtin)) extern const blockDim;
dim3 __attribute__((device_builtin)) extern const gridDim;
int __attribute__((device_builtin)) extern const warpSize;




}
# 120 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 2
# 201 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaLaunchKernel(
  const T *func,
  dim3 gridDim,
  dim3 blockDim,
  void **args,
  size_t sharedMem = 0,
  cudaStream_t stream = 0
)
{
    return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream);
}
# 263 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaLaunchCooperativeKernel(
  const T *func,
  dim3 gridDim,
  dim3 blockDim,
  void **args,
  size_t sharedMem = 0,
  cudaStream_t stream = 0
)
{
    return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream);
}
# 307 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
static __inline__ __attribute__((host)) cudaError_t cudaEventCreate(
  cudaEvent_t *event,
  unsigned int flags
)
{
  return ::cudaEventCreateWithFlags(event, flags);
}
# 372 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
static __inline__ __attribute__((host)) cudaError_t cudaMallocHost(
  void **ptr,
  size_t size,
  unsigned int flags
)
{
  return ::cudaHostAlloc(ptr, size, flags);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaHostAlloc(
  T **ptr,
  size_t size,
  unsigned int flags
)
{
  return ::cudaHostAlloc((void**)(void*)ptr, size, flags);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaHostGetDevicePointer(
  T **pDevice,
  void *pHost,
  unsigned int flags
)
{
  return ::cudaHostGetDevicePointer((void**)(void*)pDevice, pHost, flags);
}
# 501 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMallocManaged(
  T **devPtr,
  size_t size,
  unsigned int flags = 0x01
)
{
  return ::cudaMallocManaged((void**)(void*)devPtr, size, flags);
}
# 591 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaStreamAttachMemAsync(
  cudaStream_t stream,
  T *devPtr,
  size_t length = 0,
  unsigned int flags = 0x04
)
{
  return ::cudaStreamAttachMemAsync(stream, (void*)devPtr, length, flags);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMalloc(
  T **devPtr,
  size_t size
)
{
  return ::cudaMalloc((void**)(void*)devPtr, size);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMallocHost(
  T **ptr,
  size_t size,
  unsigned int flags = 0
)
{
  return cudaMallocHost((void**)(void*)ptr, size, flags);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMallocPitch(
  T **devPtr,
  size_t *pitch,
  size_t width,
  size_t height
)
{
  return ::cudaMallocPitch((void**)(void*)devPtr, pitch, width, height);
}
# 640 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
static __inline__ __attribute__((host)) cudaError_t cudaMallocAsync(
  void **ptr,
  size_t size,
  cudaMemPool_t memPool,
  cudaStream_t stream
)
{
  return ::cudaMallocFromPoolAsync(ptr, size, memPool, stream);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMallocAsync(
  T **ptr,
  size_t size,
  cudaMemPool_t memPool,
  cudaStream_t stream
)
{
  return ::cudaMallocFromPoolAsync((void**)(void*)ptr, size, memPool, stream);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMallocAsync(
  T **ptr,
  size_t size,
  cudaStream_t stream
)
{
  return ::cudaMallocAsync((void**)(void*)ptr, size, stream);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMallocFromPoolAsync(
  T **ptr,
  size_t size,
  cudaMemPool_t memPool,
  cudaStream_t stream
)
{
  return ::cudaMallocFromPoolAsync((void**)(void*)ptr, size, memPool, stream);
}
# 719 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMemcpyToSymbol(
  const T &symbol,
  const void *src,
        size_t count,
        size_t offset = 0,
        enum cudaMemcpyKind kind = cudaMemcpyHostToDevice
)
{
  return ::cudaMemcpyToSymbol((const void*)&symbol, src, count, offset, kind);
}
# 773 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMemcpyToSymbolAsync(
  const T &symbol,
  const void *src,
        size_t count,
        size_t offset = 0,
        enum cudaMemcpyKind kind = cudaMemcpyHostToDevice,
        cudaStream_t stream = 0
)
{
  return ::cudaMemcpyToSymbolAsync((const void*)&symbol, src, count, offset, kind, stream);
}
# 821 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMemcpyFromSymbol(
        void *dst,
  const T &symbol,
        size_t count,
        size_t offset = 0,
        enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost
)
{
  return ::cudaMemcpyFromSymbol(dst, (const void*)&symbol, count, offset, kind);
}
# 875 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaMemcpyFromSymbolAsync(
        void *dst,
  const T &symbol,
        size_t count,
        size_t offset = 0,
        enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost,
        cudaStream_t stream = 0
)
{
  return ::cudaMemcpyFromSymbolAsync(dst, (const void*)&symbol, count, offset, kind, stream);
}
# 944 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaGraphAddMemcpyNodeToSymbol(
    cudaGraphNode_t *pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies,
    size_t numDependencies,
    const T &symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
  return ::cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, (const void*)&symbol, src, count, offset, kind);
}
# 1015 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaGraphAddMemcpyNodeFromSymbol(
    cudaGraphNode_t* pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies,
    size_t numDependencies,
    void* dst,
    const T &symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
  return ::cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, (const void*)&symbol, count, offset, kind);
}
# 1066 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(
    cudaGraphNode_t node,
    const T &symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
  return ::cudaGraphMemcpyNodeSetParamsToSymbol(node, (const void*)&symbol, src, count, offset, kind);
}
# 1114 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(
    cudaGraphNode_t node,
    void* dst,
    const T &symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
  return ::cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, (const void*)&symbol, count, offset, kind);
}
# 1172 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    const T &symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
    return ::cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, (const void*)&symbol, src, count, offset, kind);
}
# 1231 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    void* dst,
    const T &symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
  return ::cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, (const void*)&symbol, count, offset, kind);
}
# 1267 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaGetSymbolAddress(
        void **devPtr,
  const T &symbol
)
{
  return ::cudaGetSymbolAddress(devPtr, (const void*)&symbol);
}
# 1299 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaGetSymbolSize(
        size_t *size,
  const T &symbol
)
{
  return ::cudaGetSymbolSize(size, (const void*)&symbol);
}
# 1343 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaBindTexture(
        size_t *offset,
  const struct texture<T, dim, readMode> &tex,
  const void *devPtr,
  const struct cudaChannelFormatDesc &desc,
        size_t size = 
# 1349 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 3 4
                                                (0x7fffffff * 2U + 1U)

# 1350 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
)
{
  return ::cudaBindTexture(offset, &tex, devPtr, &desc, size);
}
# 1389 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaBindTexture(
        size_t *offset,
  const struct texture<T, dim, readMode> &tex,
  const void *devPtr,
        size_t size = 
# 1394 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h" 3 4
                                                (0x7fffffff * 2U + 1U)

# 1395 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
)
{
  return cudaBindTexture(offset, tex, devPtr, tex.channelDesc, size);
}
# 1446 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaBindTexture2D(
        size_t *offset,
  const struct texture<T, dim, readMode> &tex,
  const void *devPtr,
  const struct cudaChannelFormatDesc &desc,
  size_t width,
  size_t height,
  size_t pitch
)
{
  return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch);
}
# 1505 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaBindTexture2D(
        size_t *offset,
  const struct texture<T, dim, readMode> &tex,
  const void *devPtr,
  size_t width,
  size_t height,
  size_t pitch
)
{
  return ::cudaBindTexture2D(offset, &tex, devPtr, &tex.channelDesc, width, height, pitch);
}
# 1548 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaBindTextureToArray(
  const struct texture<T, dim, readMode> &tex,
  cudaArray_const_t array,
  const struct cudaChannelFormatDesc &desc
)
{
  return ::cudaBindTextureToArray(&tex, array, &desc);
}
# 1587 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaBindTextureToArray(
  const struct texture<T, dim, readMode> &tex,
  cudaArray_const_t array
)
{
  struct cudaChannelFormatDesc desc;
  cudaError_t err = ::cudaGetChannelDesc(&desc, array);

  return err == cudaSuccess ? cudaBindTextureToArray(tex, array, desc) : err;
}
# 1629 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaBindTextureToMipmappedArray(
  const struct texture<T, dim, readMode> &tex,
  cudaMipmappedArray_const_t mipmappedArray,
  const struct cudaChannelFormatDesc &desc
)
{
  return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc);
}
# 1668 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaBindTextureToMipmappedArray(
  const struct texture<T, dim, readMode> &tex,
  cudaMipmappedArray_const_t mipmappedArray
)
{
  struct cudaChannelFormatDesc desc;
  cudaArray_t levelArray;
  cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0);

  if (err != cudaSuccess) {
      return err;
  }
  err = ::cudaGetChannelDesc(&desc, levelArray);

  return err == cudaSuccess ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err;
}
# 1711 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaUnbindTexture(
  const struct texture<T, dim, readMode> &tex
)
{
  return ::cudaUnbindTexture(&tex);
}
# 1747 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim, enum cudaTextureReadMode readMode>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaGetTextureAlignmentOffset(
        size_t *offset,
  const struct texture<T, dim, readMode> &tex
)
{
  return ::cudaGetTextureAlignmentOffset(offset, &tex);
}
# 1799 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaFuncSetCacheConfig(
  T *func,
  enum cudaFuncCache cacheConfig
)
{
  return ::cudaFuncSetCacheConfig((const void*)func, cacheConfig);
}

template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaFuncSetSharedMemConfig(
  T *func,
  enum cudaSharedMemConfig config
)
{
  return ::cudaFuncSetSharedMemConfig((const void*)func, config);
}
# 1847 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks,
    T func,
    int blockSize,
    size_t dynamicSMemSize)
{
    return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void*)func, blockSize, dynamicSMemSize, 0x00);
}
# 1899 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks,
    T func,
    int blockSize,
    size_t dynamicSMemSize,
    unsigned int flags)
{
    return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void*)func, blockSize, dynamicSMemSize, flags);
}




class __cudaOccupancyB2DHelper {
  size_t n;
public:
  inline __attribute__((host)) __attribute__((device)) __cudaOccupancyB2DHelper(size_t n_) : n(n_) {}
  inline __attribute__((host)) __attribute__((device)) size_t operator()(int)
  {
      return n;
  }
};
# 1969 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<typename UnaryFunction, class T>
static __inline__ __attribute__((host)) __attribute__((device)) cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
    int *minGridSize,
    int *blockSize,
    T func,
    UnaryFunction blockSizeToDynamicSMemSize,
    int blockSizeLimit = 0,
    unsigned int flags = 0)
{
    cudaError_t status;


    int device;
    struct cudaFuncAttributes attr;


    int maxThreadsPerMultiProcessor;
    int warpSize;
    int devMaxThreadsPerBlock;
    int multiProcessorCount;
    int funcMaxThreadsPerBlock;
    int occupancyLimit;
    int granularity;


    int maxBlockSize = 0;
    int numBlocks = 0;
    int maxOccupancy = 0;


    int blockSizeToTryAligned;
    int blockSizeToTry;
    int blockSizeLimitAligned;
    int occupancyInBlocks;
    int occupancyInThreads;
    size_t dynamicSMemSize;





    if (!minGridSize || !blockSize || !func) {
        return cudaErrorInvalidValue;
    }





    status = ::cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &maxThreadsPerMultiProcessor,
        cudaDevAttrMaxThreadsPerMultiProcessor,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &warpSize,
        cudaDevAttrWarpSize,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &devMaxThreadsPerBlock,
        cudaDevAttrMaxThreadsPerBlock,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &multiProcessorCount,
        cudaDevAttrMultiProcessorCount,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaFuncGetAttributes(&attr, func);
    if (status != cudaSuccess) {
        return status;
    }

    funcMaxThreadsPerBlock = attr.maxThreadsPerBlock;





    occupancyLimit = maxThreadsPerMultiProcessor;
    granularity = warpSize;

    if (blockSizeLimit == 0) {
        blockSizeLimit = devMaxThreadsPerBlock;
    }

    if (devMaxThreadsPerBlock < blockSizeLimit) {
        blockSizeLimit = devMaxThreadsPerBlock;
    }

    if (funcMaxThreadsPerBlock < blockSizeLimit) {
        blockSizeLimit = funcMaxThreadsPerBlock;
    }

    blockSizeLimitAligned = ((blockSizeLimit + (granularity - 1)) / granularity) * granularity;

    for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) {



        if (blockSizeLimit < blockSizeToTryAligned) {
            blockSizeToTry = blockSizeLimit;
        } else {
            blockSizeToTry = blockSizeToTryAligned;
        }

        dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry);

        status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &occupancyInBlocks,
            func,
            blockSizeToTry,
            dynamicSMemSize,
            flags);

        if (status != cudaSuccess) {
            return status;
        }

        occupancyInThreads = blockSizeToTry * occupancyInBlocks;

        if (occupancyInThreads > maxOccupancy) {
            maxBlockSize = blockSizeToTry;
            numBlocks = occupancyInBlocks;
            maxOccupancy = occupancyInThreads;
        }



        if (occupancyLimit == maxOccupancy) {
            break;
        }
    }







    *minGridSize = numBlocks * multiProcessorCount;
    *blockSize = maxBlockSize;

    return status;
}
# 2165 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<typename UnaryFunction, class T>
static __inline__ __attribute__((host)) __attribute__((device)) cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMem(
    int *minGridSize,
    int *blockSize,
    T func,
    UnaryFunction blockSizeToDynamicSMemSize,
    int blockSizeLimit = 0)
{
    return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0x00);
}
# 2211 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) __attribute__((device)) cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int *minGridSize,
    int *blockSize,
    T func,
    size_t dynamicSMemSize = 0,
    int blockSizeLimit = 0)
{
  return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, __cudaOccupancyB2DHelper(dynamicSMemSize), blockSizeLimit, 0x00);
}
# 2249 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(
    size_t *dynamicSmemSize,
    T func,
    int numBlocks,
    int blockSize)
{
    return ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, (const void*)func, numBlocks, blockSize);
}
# 2308 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) __attribute__((device)) cudaError_t cudaOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize,
    int *blockSize,
    T func,
    size_t dynamicSMemSize = 0,
    int blockSizeLimit = 0,
    unsigned int flags = 0)
{
    return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, __cudaOccupancyB2DHelper(dynamicSMemSize), blockSizeLimit, flags);
}
# 2351 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaFuncGetAttributes(
  struct cudaFuncAttributes *attr,
  T *entry
)
{
  return ::cudaFuncGetAttributes(attr, (const void*)entry);
}
# 2396 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T>
static __inline__ __attribute__((host)) cudaError_t cudaFuncSetAttribute(
  T *entry,
  enum cudaFuncAttribute attr,
  int value
)
{
  return ::cudaFuncSetAttribute((const void*)entry, attr, value);
}
# 2428 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaBindSurfaceToArray(
  const struct surface<T, dim> &surf,
  cudaArray_const_t array,
  const struct cudaChannelFormatDesc &desc
)
{
  return ::cudaBindSurfaceToArray(&surf, array, &desc);
}
# 2459 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template<class T, int dim>
static __attribute__((deprecated)) __inline__ __attribute__((host)) cudaError_t cudaBindSurfaceToArray(
  const struct surface<T, dim> &surf,
  cudaArray_const_t array
)
{
  struct cudaChannelFormatDesc desc;
  cudaError_t err = ::cudaGetChannelDesc(&desc, array);

  return err == cudaSuccess ? cudaBindSurfaceToArray(surf, array, desc) : err;
}
# 2480 "/usr/local/cuda-11.2/bin/../targets/x86_64-linux/include/cuda_runtime.h"
#pragma GCC diagnostic pop
# 1 "<command-line>" 2
# 1 "abc.cu"

# 1 "/usr/include/c++/10/stdlib.h" 1 3
# 3 "abc.cu" 2

# 1 "/usr/include/assert.h" 1 3 4
# 64 "/usr/include/assert.h" 3 4

# 64 "/usr/include/assert.h" 3 4
extern "C" {


extern void __assert_fail (const char *__assertion, const char *__file,
      unsigned int __line, const char *__function)
     throw () __attribute__ ((__noreturn__));


extern void __assert_perror_fail (int __errnum, const char *__file,
      unsigned int __line, const char *__function)
     throw () __attribute__ ((__noreturn__));




extern void __assert (const char *__assertion, const char *__file, int __line)
     throw () __attribute__ ((__noreturn__));


}
# 5 "abc.cu" 2
# 1 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stdint.h" 1 3 4
# 9 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stdint.h" 3 4
# 1 "/usr/include/stdint.h" 1 3 4
# 26 "/usr/include/stdint.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 27 "/usr/include/stdint.h" 2 3 4

# 1 "/usr/include/bits/wchar.h" 1 3 4
# 29 "/usr/include/stdint.h" 2 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 30 "/usr/include/stdint.h" 2 3 4







# 1 "/usr/include/bits/stdint-uintn.h" 1 3 4
# 24 "/usr/include/bits/stdint-uintn.h" 3 4
typedef __uint8_t uint8_t;
typedef __uint16_t uint16_t;
typedef __uint32_t uint32_t;
typedef __uint64_t uint64_t;
# 38 "/usr/include/stdint.h" 2 3 4





typedef __int_least8_t int_least8_t;
typedef __int_least16_t int_least16_t;
typedef __int_least32_t int_least32_t;
typedef __int_least64_t int_least64_t;


typedef __uint_least8_t uint_least8_t;
typedef __uint_least16_t uint_least16_t;
typedef __uint_least32_t uint_least32_t;
typedef __uint_least64_t uint_least64_t;





typedef signed char int_fast8_t;

typedef long int int_fast16_t;
typedef long int int_fast32_t;
typedef long int int_fast64_t;
# 71 "/usr/include/stdint.h" 3 4
typedef unsigned char uint_fast8_t;

typedef unsigned long int uint_fast16_t;
typedef unsigned long int uint_fast32_t;
typedef unsigned long int uint_fast64_t;
# 87 "/usr/include/stdint.h" 3 4
typedef long int intptr_t;


typedef unsigned long int uintptr_t;
# 101 "/usr/include/stdint.h" 3 4
typedef __intmax_t intmax_t;
typedef __uintmax_t uintmax_t;
# 10 "/usr/lib/gcc/x86_64-redhat-linux/10/include/stdint.h" 2 3 4
# 6 "abc.cu" 2
# 18 "abc.cu"

# 18 "abc.cu"
static __attribute__((device)) __inline__ uint64_t __nano(){
  uint64_t mclk;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
  return mclk ;
}

__attribute__((global)) void foo(float* a, float* b, float* c)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < 32)
    {
        tid = tid * (4096/sizeof(float));
        c[tid] = a[tid] + b[tid];
        tid = tid + ((4096/sizeof(float)) * 32);
        c[tid] = a[tid] + b[tid];
        tid = tid + ((4096/sizeof(float)) * 32);
        c[tid] = a[tid] + b[tid];
    }

}

__attribute__((global)) void foo(float* a, float* b, float* c)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t ts1;
    size_t ts2;
    if (tid < 32)
    {
        tid = tid * (4096/sizeof(float));

        c[tid] = a[tid] + b[tid];


        tid = tid + ((4096/sizeof(float)) * 32);
        c[tid] = a[tid] + b[tid];

        tid = tid + ((4096/sizeof(float)) * 32);
        c[tid] = a[tid] + b[tid];

    }

}

size_t pad_2MB(size_t val)
{
    size_t ret = val / (4096 * 512);
    size_t diff = val % (4096 * 512);
    if (diff)
    {
        ret += (4096 * 512);
    }
    return ret;
}

int main(void)
{
    float* a;
    float* b;
    float* c;

    
# 78 "abc.cu" 3 4
   (static_cast <bool> (
# 78 "abc.cu"
   !cudaMallocManaged(&a, pad_2MB(2 * sizeof(float) * 32 * (4096/sizeof(float))))
# 78 "abc.cu" 3 4
   ) ? void (0) : __assert_fail (
# 78 "abc.cu"
   "!cudaMallocManaged(&a, pad_2MB(2 * sizeof(float) * TSIZE * FPSIZE))"
# 78 "abc.cu" 3 4
   , "abc.cu", 78, __extension__ __PRETTY_FUNCTION__))
# 78 "abc.cu"
                                                                              ;
    
# 79 "abc.cu" 3 4
   (static_cast <bool> (
# 79 "abc.cu"
   !cudaMallocManaged(&b, pad_2MB(2 * sizeof(float) * 32 * (4096/sizeof(float))))
# 79 "abc.cu" 3 4
   ) ? void (0) : __assert_fail (
# 79 "abc.cu"
   "!cudaMallocManaged(&b, pad_2MB(2 * sizeof(float) * TSIZE * FPSIZE))"
# 79 "abc.cu" 3 4
   , "abc.cu", 79, __extension__ __PRETTY_FUNCTION__))
# 79 "abc.cu"
                                                                              ;
    
# 80 "abc.cu" 3 4
   (static_cast <bool> (
# 80 "abc.cu"
   !cudaMallocManaged(&c, pad_2MB(2* sizeof(float) * 32 * (4096/sizeof(float))))
# 80 "abc.cu" 3 4
   ) ? void (0) : __assert_fail (
# 80 "abc.cu"
   "!cudaMallocManaged(&c, pad_2MB(2* sizeof(float) * TSIZE * FPSIZE))"
# 80 "abc.cu" 3 4
   , "abc.cu", 80, __extension__ __PRETTY_FUNCTION__))
# 80 "abc.cu"
                                                                             ;





    for (size_t i = 0; i < 32 * (4096/sizeof(float)); i++)
    {
        a[i] = i;
        b[i] = i;
        c[i] = i;
    }

    foo<<<32/32, 32>>>(a, b, c);
    cudaDeviceSynchronize();
    return 0;
}