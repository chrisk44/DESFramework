#ifndef UTILITIES_H
#define UTILITIES_H

#include "cuda_runtime.h"
#include <cuda.h>

#define cce() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

#define BLOCK_SIZE 1024
#define MAX_DIMENSIONS 10
#define min(X, Y)  ((X) < (Y) ? (X) : (Y))
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

//#define DEBUG

#define TAG_READY 0
#define TAG_DATA_COUNT 1
#define TAG_DATA 2
#define TAG_RESULTS 3
#define TAG_ID 4

class Model {
public:
	// float* point will be a D-dimension vector
	__host__   virtual bool validate_cpu(float* point) = 0;
	__device__ virtual bool validate_gpu(float point[]) = 0;

	virtual bool toBool() = 0;
};

struct Limit {
	float lowerLimit;
	float upperLimit;
	unsigned long N;
};

#endif