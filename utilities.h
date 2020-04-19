#ifndef UTILITIES_H
#define UTILITIES_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <semaphore.h>
#include <mpi.h>

#define cce() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

// Slow start parameters for batch size increment
#define SS_THRESHOLD 200000000
#define SS_STEP 1000

// Memory parameters
#define MEM_CPU_SPARE_BYTES 100*1024*1024
#define MEM_GPU_SPARE_BYTES 100*1024*1024

// Computing parameters
#define BLOCK_SIZE 1024
#define COMPUTE_BATCH_SIZE 500
#define NUM_OF_STREAMS 8
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
#define DEBUG 1

// MPI
#define RECV_SLEEP_MS 1
#define TAG_READY 0
#define TAG_DATA_COUNT 1
#define TAG_DATA 2
#define TAG_RESULTS 3
#define TAG_MAX_DATA_COUNT 4
#define TAG_EXITING 5
#define TAG_RESULTS_DATA 6

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
    timespec t1, t2;
    bool started = false, stopped = false;

public:
    void start();
    void stop();
    void reset();
    float getNsec();
    float getMsec();
};

unsigned long getDefaultCPUBatchSize();
unsigned long getDefaultGPUBatchSize();

// MPI_Recv without busy wait
void MMPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status);

struct Limit {
	DATA_TYPE lowerLimit;
	DATA_TYPE upperLimit;
	unsigned long N;
    DATA_TYPE step;
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
	// ...
};

struct ComputeProcessStatus {
    // These are allocated with malloc so they are not initialized
    // Initialization should be done by masterThread()
	unsigned long maxBatchSize;
	unsigned long currentBatchSize;
    unsigned long computingIndex;
    int assignedElements;
	unsigned int jobsCompleted;
	unsigned elementsCalculated;
	bool finished;
	bool initialized;

	Stopwatch stopwatch;
};

struct ProcessingThreadInfo{
    int id;
    int numOfElements;
    unsigned long* startPointIdx;
    RESULT_TYPE* results;
    sem_t semData;
    sem_t* semResults;

    // Data distribution
    Stopwatch stopwatch;
    float ratio;
};

#endif
