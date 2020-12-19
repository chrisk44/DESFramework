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
#define MAX_DIMENSIONS 30

#define RESULT_TYPE float
#define DATA_TYPE double
#define RESULT_MPI_TYPE MPI_FLOAT
#define DATA_MPI_TYPE MPI_DOUBLE

// Debugging
// #define DBG_START_STOP      // Messages about starting/stopping processes and threads
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
#define TAG_READY 1
#define TAG_DATA 2
#define TAG_RESULTS 3
#define TAG_MAX_DATA_COUNT 4
#define TAG_EXITING 5
#define TAG_RESULTS_DATA 6

typedef RESULT_TYPE (*validationFunc_t)(DATA_TYPE*, void*);
typedef bool (*toBool_t)(RESULT_TYPE);

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
void asd();

// MPI_Recv without busy wait
void MMPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status);

struct Limit {
	DATA_TYPE lowerLimit;
	DATA_TYPE upperLimit;
	unsigned int N;
    DATA_TYPE step;
};

struct AssignedWork {
    unsigned long startPoint;
    unsigned long numOfElements;
};

struct ThreadCommonData {
    sem_t semResults;
    int listIndex;
    unsigned long currentBatchStart;
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
	unsigned long batchSize;
	ProcessingType processingType = PROCESSING_TYPE_BOTH;
    ResultSaveType resultSaveType = SAVE_TYPE_ALL;
    char* saveFile = nullptr;
    bool threadBalancing = true;
    bool slaveBalancing = true;
	bool benchmark = false;
    void* dataPtr = nullptr;
    unsigned long dataSize = 0;
    bool overrideMemoryRestrictions = false;
    int blockSize = 256;
    int computeBatchSize = 200;
    int gpuStreams = 8;
    int cpuComputeBatchSize = 10000;
    bool cpuDynamicScheduling = true;
    bool threadBalancingAverage = false;
    unsigned long slowStartBase = 5000000;
    int slowStartLimit = 3;
    int minMsForRatioAdjustment = 0;
    bool finalizeAfterExecution = true;
    bool printProgress = true;
};

struct SlaveProcessInfo {
    // These are allocated with malloc so they are not initialized
    // Initialization should be done by masterProcess()
    int id;
	unsigned long maxBatchSize;
	unsigned long currentBatchSize;
    AssignedWork work;
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
    unsigned long numOfElements;
    unsigned long startPoint;
    RESULT_TYPE* results;
    sem_t semData;

    Stopwatch stopwatch;
    float ratio;
    float totalRatio;
};

#endif
