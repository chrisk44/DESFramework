#ifndef UTILITIES_H
#define UTILITIES_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <limits.h>
#include <mpi.h>
#include <semaphore.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>

#define cce() {                                          \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

#define fatal(e){                                        \
    printf("Error: %s:%d: %s\n", __FILE__, __LINE__, e); \
    exit(3);                                             \
}

// Memory parameters
#define MEM_GPU_SPARE_BYTES 100*1024*1024

// Computing parameters
#define MAX_DIMENSIONS 20       // TODO: Calculate this accurately

// Batch size slow-start and ratio limits
#define SLOW_START_BATCH_SIZE_BASE 1000000    // BASE * (2^LIMIT) MUST FIT IN AN UNSIGNED LONG
#define SLOW_START_LIMIT 6
#define MIN_MS_FOR_RATIO_ADJUSTMENT 0

#define RESULT_TYPE float
#define DATA_TYPE double
#define RESULT_MPI_TYPE MPI_FLOAT
#define DATA_MPI_TYPE MPI_DOUBLE

// Debugging
#define DBG_START_STOP      // Messages about starting/stopping processes and threads
// #define DBG_QUEUE           // Messages about queueing work (coordinator->worker threads, worker->gpu streams)
// #define DBG_MPI_STEPS       // Messages after each MPI step
// #define DBG_RATIO           // Messages about changes in ratios (masterProcess and coordinatorThread)
// #define DBG_DATA            // Messages about the exact data being assigned (start points)
// #define DBG_MEMORY          // Messages about memory management (addresses, reallocations)
// #define DBG_RESULTS         // Messages with the exact results being passed around
// #define DBG_TIME            // Print time measuraments for various parts of the code
#define DBG_SNH             // Should not happen

// MPI
#define RECV_SLEEP_US 100     // Time in micro-seconds to sleep between checking for data in MPI_Recv
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
	inline __host__   virtual RESULT_TYPE validate_cpu(DATA_TYPE* point, void* dataPtr) = 0;
	inline __device__ virtual RESULT_TYPE validate_gpu(DATA_TYPE* point, void* dataPtr) = 0;

	inline __host__ __device__ virtual bool toBool(RESULT_TYPE result) = 0;
};

class Stopwatch{
private:
    timespec t1, t2;

public:
    void start();
    void stop();
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
	PROCESSING_TYPE_CPU,
	PROCESSING_TYPE_GPU,
	PROCESSING_TYPE_BOTH
};

enum ResultSaveType {
    SAVE_TYPE_ALL,
    SAVE_TYPE_LIST
};

struct ParallelFrameworkParameters {
	unsigned int D;
	unsigned int batchSize;
	ProcessingType processingType = PROCESSING_TYPE_BOTH;
    ResultSaveType resultSaveType = SAVE_TYPE_ALL;
    char* saveFile = nullptr;
    bool threadBalancing = true;
    bool slaveBalancing = true;
	bool benchmark = false;
    void* dataPtr = nullptr;
    unsigned long dataSize = 0;
    int blockSize = 1024;
    int computeBatchSize = 1000;
    int gpuStreams = 8;
	// ...
};

struct SlaveProcessInfo {
    // These are allocated with malloc so they are not initialized
    // Initialization should be done by masterProcess()
    int id;
	unsigned long maxBatchSize;
	unsigned long currentBatchSize;
    unsigned long computingIndex;
    int assignedElements;
	unsigned int jobsCompleted;
	unsigned elementsCalculated;
	bool finished;

	Stopwatch stopwatch;
    float lastScore;
    float lastAssignedElements;
    float ratio;
};

struct ComputeThreadInfo{
    int id;
    int numOfElements;
    unsigned long* startPointIdx;
    RESULT_TYPE* results;
    sem_t semData;
    sem_t* semResults;
    int* listIndexPtr;

    Stopwatch stopwatch;
    float ratio;
};

#endif
