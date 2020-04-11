#ifndef UTILITIES_H
#define UTILITIES_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>

#define cce() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

// Server parameters
#define DEFAULT_PORT "9000"

// Slow start parameters for batch size increment
#define SS_THRESHOLD 200000000
#define SS_STEP 1000

// Memory parameters
#define MEM_CPU_SPARE_BYTES 100*1024*1024
#define MEM_GPU_SPARE_BYTES 100*1024*1024

// Computing parameters
#define BLOCK_SIZE 1024
#define MAX_DIMENSIONS 10
//#define min(X, Y)  ((X) < (Y) ? (X) : (Y))
//#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

#define RESULT_TYPE float
#define DATA_TYPE double
#define RESULT_MPI_TYPE MPI_FLOAT

// 0: Only errors
// 1: Processes start/stop
// 2: Processes steps
// 3: Data transfers
// 4: Data calculations
#define DEBUG 2

#define TAG_READY 0
#define TAG_DATA_COUNT 1
#define TAG_DATA 2
#define TAG_RESULTS 3
#define TAG_MAX_DATA_COUNT 4

class Model {
public:
	// float* point will be a D-dimension vector
	__host__   virtual RESULT_TYPE validate_cpu(DATA_TYPE* point) = 0;
	__device__ virtual RESULT_TYPE validate_gpu(DATA_TYPE* point) = 0;

	virtual bool toBool() = 0;
};

class Stopwatch{
    // Measures time is nano seconds
private:
    struct timespec t1, t2;

public:
    void start();
    void stop();
    float getNsec();
    float getMsec();
};

long getDefaultCPUBatchSize();
long getDefaultGPUBatchSize();

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
	unsigned int batchSize;
	ProcessingType processingType = TYPE_BOTH;
	bool dynamicBatchSize = true;
	bool benchmark = false;
	bool remote = false;
	std::string serverName;
	// ...
};

struct ComputeProcessStatus {
	unsigned long computingIndex = 0;
	unsigned long maxBatchSize;
	unsigned long currentBatchSize;		// Initialized by masterThread()
	unsigned int jobsCompleted = 0;
	unsigned elementsCalculated = 0;
	bool finished = false;
	bool initialized = false;

	Stopwatch stopwatch;
};

#endif
