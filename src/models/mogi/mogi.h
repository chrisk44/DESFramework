#pragma once

#include <cuda_runtime.h>

namespace mogi {

__host__ __device__ float doValidateM1(double* x, void* dataPtr);
__host__ __device__ float doValidateM2(double* x, void* dataPtr);

}
