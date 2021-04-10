#include "../../shared.h"
#include "cuknl_shared.h"

void check_errors(int line_num, const char* file)
{
    cudaDeviceSynchronize();

    int result = cudaGetLastError();

    if(result != cudaSuccess)
    {
        die(line_num, file, "Error in %s - return code %d (%s)\n",
                file, result, cuda_codes(result));
    }
}

// Enumeration for the set of potential CUDA error codes.
const char* cuda_codes(int code)
{
	switch(code)
	{
		case cudaSuccess: return "cudaSuccess"; // 0
		case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration"; // 1
		case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation"; // 2
		case cudaErrorInitializationError: return "cudaErrorInitializationError"; // 3
		case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure"; // 4
		case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure"; // 5
		case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout"; // 6
		case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources"; // 7
		case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction"; // 8
		case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration"; // 9
		case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice"; // 10
		case cudaErrorInvalidValue: return "cudaErrorInvalidValue";// 11
		case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";// 12
		case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";// 13
		case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed";// 14
		case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed";// 15
		case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";// 16
		case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";// 17
		case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";// 18
		case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";// 19
		case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";// 20
		case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";// 21
		case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";// 22
		case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";// 23
		case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";// 24
		case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";// 25
		case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";// 26
		case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";// 27
		case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";// 28
		case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading";// 29
		case cudaErrorUnknown: return "cudaErrorUnknown";// 30
		case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";// 31
		case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";// 32
		case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";// 33
		case cudaErrorNotReady: return "cudaErrorNotReady";// 34
		case cudaErrorInsufficientDriver: return "cudaErrorInsufficientDriver";// 35
		case cudaErrorSetOnActiveProcess: return "cudaErrorSetOnActiveProcess";// 36
		case cudaErrorInvalidSurface: return "cudaErrorInvalidSurface";// 37
		case cudaErrorNoDevice: return "cudaErrorNoDevice";// 38
		case cudaErrorECCUncorrectable: return "cudaErrorECCUncorrectable";// 39
		case cudaErrorSharedObjectSymbolNotFound: return "cudaErrorSharedObjectSymbolNotFound";// 40
		case cudaErrorSharedObjectInitFailed: return "cudaErrorSharedObjectInitFailed";// 41
		case cudaErrorUnsupportedLimit: return "cudaErrorUnsupportedLimit";// 42
		case cudaErrorDuplicateVariableName: return "cudaErrorDuplicateVariableName";// 43
		case cudaErrorDuplicateTextureName: return "cudaErrorDuplicateTextureName";// 44
		case cudaErrorDuplicateSurfaceName: return "cudaErrorDuplicateSurfaceName";// 45
		case cudaErrorDevicesUnavailable: return "cudaErrorDevicesUnavailable";// 46
		case cudaErrorInvalidKernelImage: return "cudaErrorInvalidKernelImage";// 47
		case cudaErrorNoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice";// 48
		case cudaErrorIncompatibleDriverContext: return "cudaErrorIncompatibleDriverContext";// 49
		case cudaErrorPeerAccessAlreadyEnabled: return "cudaErrorPeerAccessAlreadyEnabled";// 50
		case cudaErrorPeerAccessNotEnabled: return "cudaErrorPeerAccessNotEnabled";// 51
		case cudaErrorDeviceAlreadyInUse: return "cudaErrorDeviceAlreadyInUse";// 52
		case cudaErrorProfilerDisabled: return "cudaErrorProfilerDisabled";// 53
		case cudaErrorProfilerNotInitialized: return "cudaErrorProfilerNotInitialized";// 54
		case cudaErrorProfilerAlreadyStarted: return "cudaErrorProfilerAlreadyStarted";// 55
		case cudaErrorProfilerAlreadyStopped: return "cudaErrorProfilerAlreadyStopped";// 56
		case cudaErrorAssert: return "cudaErrorAssert";// 57
		case cudaErrorTooManyPeers: return "cudaErrorTooManyPeers";// 58
		case cudaErrorHostMemoryAlreadyRegistered: return "cudaErrorHostMemoryAlreadyRegistered";// 59
		case cudaErrorHostMemoryNotRegistered: return "cudaErrorHostMemoryNotRegistered";// 60
		case cudaErrorOperatingSystem: return "cudaErrorOperatingSystem";// 61
		case cudaErrorStartupFailure: return "cudaErrorStartupFailure";// 62
		case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";// 63
		default: return "Unknown error";
	}
}


