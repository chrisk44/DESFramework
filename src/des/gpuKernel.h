#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "types.h"

#define MAX_CONSTANT_MEMORY (65536 - 24)			// Don't know why...

namespace desf {

#ifndef DES_MONILITHIC
extern __device__ __constant__ char constantMemoryPtr[MAX_CONSTANT_MEMORY];
void cudaMemcpyToSymbolWrapper(const void* src, size_t count, size_t offset);
#else
__device__ __constant__ char constantMemoryPtr[MAX_CONSTANT_MEMORY];

static void cudaMemcpyToSymbolWrapper(const void* src, size_t count, size_t offset){
    cudaMemcpyToSymbol(constantMemoryPtr, src, count, offset, cudaMemcpyHostToDevice);
}
#endif

// CUDA kernel to run the computation
__global__ void validate_kernel(validationFunc_t validationFunc, toBool_t toBool, void* results, unsigned long startingPointLinearIndex,
    const unsigned int D, const unsigned long numOfElements, const unsigned long offset, void* dataPtr,
    int dataSize, bool useSharedMemoryForData, bool useConstantMemoryForData, int* listIndexPtr,
    const int computeBatchSize);

// CUDA kernel to retrieve the address of a __device__ symbol
template<typename T, T symbol>
__global__ void getPointerFromSymbol(T* dst) {
    *dst = symbol;
}

}
