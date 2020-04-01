#pragma once

#include "cuda_runtime.h"
class Model{
public:
    __host__   virtual bool validate_cpu(float* point) = 0;
	__device__ virtual bool validate_gpu(float* point) = 0;

    virtual bool toBool() = 0;
};

