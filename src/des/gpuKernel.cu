#ifndef DES_DIRECT_COMPILATION

#include <cuda.h>
#include <cuda_runtime.h>

#include "gpuKernel.h"

__device__ __constant__ char constantMemoryPtr[MAX_CONSTANT_MEMORY];

void cudaMemcpyToSymbolWrapper(const void* src, size_t count, size_t offset){
    cudaMemcpyToSymbol(constantMemoryPtr, src, count, offset, cudaMemcpyHostToDevice);
}

#endif
