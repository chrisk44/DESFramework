#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <semaphore.h>
#include <string>

#include "definitions.h"
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

struct DesConfig {
    ProcessingType processingType = PROCESSING_TYPE_BOTH;
    ResultSaveType resultSaveType = SAVE_TYPE_LIST;
    bool finalizeAfterExecution = true;
    bool printProgress = true;
    bool benchmark = false;

    struct {
        bool overrideMemoryRestrictions = false;
        std::string saveFile;
    } output;

    bool threadBalancing = true;
    bool slaveBalancing = true;
    bool slaveDynamicScheduling = false;
    bool threadBalancingAverage = false;

    unsigned long batchSize;
    unsigned long slaveBatchSize;

    unsigned long slowStartBase = 5000000;
    int slowStartLimit = 3;
    int minMsForRatioAdjustment = 0;

    struct {
        unsigned int D;
        void* dataPtr = nullptr;
        unsigned long dataSize = 0;
    } model;

    struct {
        bool dynamicScheduling = true;
        int computeBatchSize = 10000;

        validationFunc_t forwardModel = nullptr;
        toBool_t objective = nullptr;
    } cpu;

    struct {
        int computeBatchSize = 200;
        int blockSize = 256;
        int streams = 8;

        std::map<int, validationFunc_t> forwardModels;
        std::map<int, toBool_t> objectives;
    } gpu;
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
