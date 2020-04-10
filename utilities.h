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

#define RESULT_TYPE float
#define RESULT_MPI_TYPE MPI_FLOAT

#define DATA_TYPE double


#define BLOCK_SIZE 1024
#define MAX_DIMENSIONS 10
#define min(X, Y)  ((X) < (Y) ? (X) : (Y))
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

// 0: Only errors
// 1: Processes start/stop
// 2: Processes steps
// 3: Data transfers
// 4: Data calculations
#define DEBUG 3

#define TAG_READY 0
#define TAG_DATA_COUNT 1
#define TAG_DATA 2
#define TAG_RESULTS 3

class Model {
public:
	// float* point will be a D-dimension vector
	__host__   virtual RESULT_TYPE validate_cpu(DATA_TYPE* point) = 0;
	__device__ virtual RESULT_TYPE validate_gpu(DATA_TYPE* point) = 0;

	virtual bool toBool() = 0;
};

struct Limit {
	DATA_TYPE lowerLimit;
	DATA_TYPE upperLimit;
	unsigned long N;
};

enum ProcessingType {
	TYPE_CPU,
	TYPE_GPU,
	TYPE_BOTH
};

struct ParallelFrameworkParameters {
	unsigned int D;
	unsigned int computeBatchSize;
	unsigned int batchSize;
	ProcessingType processingType;
	bool dynamicBatchSize;
	// ...
};

struct ComputeProcessDetails {
	unsigned long computingIndex = 0;
	unsigned long currentBatchSize;		// Initialized by masterThread()
	unsigned int jobsCompleted = 0;
	unsigned elementsCalculated = 0;
	bool finished = false;
	bool initialized = false;

	time_t computeStartTime;
	time_t lastTimePerElement;
};

#endif
