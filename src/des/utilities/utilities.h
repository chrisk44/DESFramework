#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <nvml.h>
#include <limits.h>
#include <mpi.h>
#include <mutex>
#include <semaphore.h>
#include <string>
#include <sys/sysinfo.h>
#include <unistd.h>

#include "stopwatch.h"

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
#ifndef MEM_GPU_SPARE_BYTES
    #define MEM_GPU_SPARE_BYTES 100*1024*1024
#endif

// Computing parameters
#define MAX_DIMENSIONS 30

#ifndef RESULT_TYPE
    #define RESULT_TYPE float
#endif
#ifndef DATA_TYPE
    #define DATA_TYPE double
#endif
#ifndef RESULT_MPI_TYPE
    #define RESULT_MPI_TYPE MPI_FLOAT
#endif
#ifndef DATA_MPI_TYPE
    #define DATA_MPI_TYPE MPI_DOUBLE
#endif

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

unsigned long getMaxCPUBytes();
unsigned long getMaxGPUBytes();
unsigned long getMaxGPUBytesForGpu(int id);
int getCpuStats(float* uptime, float* idleTime);

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

enum ProcessingType {
    PROCESSING_TYPE_CPU,
    PROCESSING_TYPE_GPU,
    PROCESSING_TYPE_BOTH
};

enum ResultSaveType {
    SAVE_TYPE_ALL,
    SAVE_TYPE_LIST
};

enum WorkerThreadType {
    CPU,
    GPU
};

struct ParallelFrameworkParameters {
    unsigned int D;
    ProcessingType processingType = PROCESSING_TYPE_BOTH;
    ResultSaveType resultSaveType = SAVE_TYPE_LIST;
    bool overrideMemoryRestrictions = false;
    bool finalizeAfterExecution = true;
    bool printProgress = true;
    bool benchmark = false;

    std::string saveFile;
    void* dataPtr = nullptr;
    unsigned long dataSize = 0;

    bool threadBalancing = true;
    bool slaveBalancing = true;
    bool slaveDynamicScheduling = false;
    bool cpuDynamicScheduling = true;
    bool threadBalancingAverage = false;

    unsigned long batchSize;
    unsigned long slaveBatchSize;
    int computeBatchSize = 200;
    int cpuComputeBatchSize = 10000;

    int blockSize = 256;
    int gpuStreams = 8;

    unsigned long slowStartBase = 5000000;
    int slowStartLimit = 3;
    int minMsForRatioAdjustment = 0;
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

class ThreadCommonData {
public:
    std::mutex syncMutex;

    int listIndex;
    unsigned long currentBatchStart;
    RESULT_TYPE* results;
    unsigned long globalFirst;
    unsigned long globalLast;
    unsigned long globalBatchStart;

    ThreadCommonData()
    {
        sem_init(&resultsSem, 0, 0);
    }

    ~ThreadCommonData()
    {
        sem_destroy(&resultsSem);
    }

    void postResultsSemaphore(){ sem_post(&resultsSem); }
    void waitResultsSemaphore(){ sem_wait(&resultsSem); }

private:
    sem_t resultsSem;

};

class ComputeThreadInfo {
public:
    int id;
    std::string name;
    WorkerThreadType type;

    unsigned long batchSize;
    unsigned long elementsCalculated;

    Stopwatch stopwatch;
    float ratio;
    float totalRatio;

    float averageUtilization = -1;
    float idleTime = 0;
    Stopwatch masterStopwatch;

    ComputeThreadInfo(int _id, std::string _name, int initSemValue, int numOfThreads)
        : id(_id),
          name(std::move(_name)),
          ratio(1.f / (float)numOfThreads),
          totalRatio(0)
    {
        sem_init(&semStart, 0, initSemValue);
    }

    ~ComputeThreadInfo() {
        sem_destroy(&semStart);
    }

    void postStartSemaphore(){ sem_post(&semStart); }
    void waitStartSemaphore(){ sem_wait(&semStart); }

private:
    sem_t semStart;

};
