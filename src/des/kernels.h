#include <cuda.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "utilities.h"

#define MAX_CONSTANT_MEMORY 1024 //(65536 - 24)			// Don't know why...
__device__ __constant__ static char constantMemoryPtr[MAX_CONSTANT_MEMORY];

void cudaMemcpyToSymbolWrapper(const void* src, size_t count, size_t offset);

// CUDA kernel to run the computation
__global__ void validate_kernel(validationFunc_t validationFunc, toBool_t toBool, RESULT_TYPE* results, unsigned long startingPointLinearIndex,
	const unsigned int D, const unsigned long numOfElements, const unsigned long offset, void* dataPtr,
	int dataSize, bool useSharedMemoryForData, bool useConstantMemoryForData, int* listIndexPtr,
    const int computeBatchSize);

// CPU kernel to run the computation
void cpu_kernel(validationFunc_t validationFunc, toBool_t toBool, RESULT_TYPE* results, Limit* limits, unsigned int D, unsigned long numOfElements, void* dataPtr, int* listIndexPtr,
    unsigned long long* idxSteps, unsigned long startingPointLinearIndex, bool dynamicScheduling, int batchSize);
