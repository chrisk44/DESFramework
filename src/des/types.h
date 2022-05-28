#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <semaphore.h>
#include <string>
#include <vector>

#include "definitions.h"
#include "stopwatch.h"

namespace desf {

template<typename node_id_t>
class Scheduler;

typedef int ComputeThreadID;

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

    AssignedWork(size_t startPoint, size_t count)
        : startPoint(startPoint),
          numOfElements(count)
    {}

    AssignedWork()
        : AssignedWork(0, 0)
    {}
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

struct GpuConfig {
    int computeBatchSize = 200;
    int blockSize = 256;
    int streams = 8;

    validationFunc_t forwardModel = nullptr;
    toBool_t objective = nullptr;
};

struct CpuConfig {
    bool dynamicScheduling = true;
    int computeBatchSize = 10000;

    validationFunc_t forwardModel = nullptr;
    toBool_t objective = nullptr;
};

class DesConfig {
public:
    DesConfig(bool deleteSchedulersWhenDestructed);
    ~DesConfig();
    ProcessingType processingType = PROCESSING_TYPE_BOTH;
    ResultSaveType resultSaveType = SAVE_TYPE_LIST;
    bool handleMPI = true;
    bool printProgress = true;
    bool benchmark = false;

    std::vector<Limit> limits;

    struct {
        bool overrideMemoryRestrictions = false;
        std::string saveFile;
    } output;

    struct {
        unsigned int D;
        void* dataPtr = nullptr;
        unsigned long dataSize = 0;
    } model;

    CpuConfig cpu;
    std::map<int, GpuConfig> gpu;

    Scheduler<int> *interNodeScheduler;
    Scheduler<int> *intraNodeScheduler;

private:
    bool m_deleteSchedulers;
};

struct SlaveProcessInfo {
    int id;
    unsigned long maxBatchSize;
    AssignedWork work;
    unsigned int jobsCompleted;
    unsigned elementsCalculated;
    bool finished;

    Stopwatch stopwatch;
};

}
