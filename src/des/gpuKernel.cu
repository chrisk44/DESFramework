#include <cuda.h>
#include <cuda_runtime.h>

#include "gpuKernel.h"
#include "utilities.h"

__constant__ char constantMemoryPtr[MAX_CONSTANT_MEMORY];

void cudaMemcpyToSymbolWrapper(const void* src, size_t count, size_t offset){
    cudaMemcpyToSymbol(constantMemoryPtr, src, count, offset, cudaMemcpyHostToDevice);
}

