#pragma once

#include <atomic>
#include <mutex>
#include <vector>

#include "computeThread.h"

namespace desf {

class CoordinatorThread {
public:
    CoordinatorThread(const DesConfig& config, std::vector<ComputeThread>& computeThreads);
    ~CoordinatorThread();

    void run();

private:
    static unsigned long calculateMaxBatchSize(const DesConfig& config);
    static unsigned long calculateMaxCpuBatchSize(const DesConfig& config);

    void log(const char* text, ...);

    std::mutex m_syncMutex;
    const DesConfig& m_config;
    std::vector<ComputeThread>& m_threads;
    int m_rank;

    unsigned long m_maxBatchSize;
    unsigned long m_maxCpuBatchSize;

};


}
