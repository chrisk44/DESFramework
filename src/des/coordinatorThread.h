#pragma once

#include <atomic>
#include <mutex>
#include <vector>

#include "computeThread.h"

namespace desf {

class CoordinatorThread {
public:
    CoordinatorThread(const DesConfig& config);
    ~CoordinatorThread();

    void run(std::vector<ComputeThread>& computeThreads);

private:
    static unsigned long calculateMaxBatchSize(const DesConfig& config);
    static unsigned long calculateMaxCpuBatchSize(const DesConfig& config);

    void log(const char* text, ...);

    std::map<ComputeThreadID, size_t> m_batchSizes;
    AssignedWork getWork(ComputeThreadID threadID);

    std::mutex m_syncMutex;
    const DesConfig& m_config;
    int m_rank;

    unsigned long m_maxBatchSize;
    unsigned long m_maxCpuBatchSize;

    int m_listIndex;
    unsigned long m_currentBatchStart;
    std::vector<RESULT_TYPE> m_results;
    unsigned long m_globalFirst;
    unsigned long m_globalLast;
    std::atomic_size_t m_globalBatchStart;
};


}
