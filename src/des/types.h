#pragma once

#include <atomic>
#include <mutex>
#include <semaphore.h>
#include <string>

#include "defs.h"
#include "stopwatch.h"

typedef RESULT_TYPE (*validationFunc_t)(DATA_TYPE*, void*);
typedef bool (*toBool_t)(RESULT_TYPE);

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
    std::atomic_size_t globalBatchStart;
};
