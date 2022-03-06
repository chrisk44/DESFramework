#pragma once

#include <cuda_runtime.h>

namespace okada {

__host__ __device__ float doValidateO1(double* x, void* dataPtr);
__host__ __device__ float doValidateO2(double* x, void* dataPtr);
__host__ __device__ float doValidateO3(double* x, void* dataPtr);

}
