#pragma once

#include <functional>
#include <string>
#include <semaphore.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>

#include <cuda_runtime.h>
#include <nvml.h>

#include "definitions.h"
#include "stopwatch.h"
#include "types.h"

namespace desf {

enum WorkerThreadType {
    CPU,
    GPU
};

typedef std::function<AssignedWork(ComputeThreadID)> WorkDispatcher;

class DesFramework;

class ComputeThread {
public:
    ComputeThread(ComputeThreadID id,
                  std::string name,
                  WorkerThreadType type,
                  const DesConfig& config,
                  const std::vector<unsigned long long>& indexSteps);

    // Dummy copy constructor which does not actually copy the whole object, just recreates it in the new location
    ComputeThread(const ComputeThread& other)
        : ComputeThread(other.m_id, other.m_name, other.m_type, other.m_config, other.m_indexSteps)
    {}

    ~ComputeThread();

    void dispatch(WorkDispatcher workDispatcher, RESULT_TYPE* results, size_t indexOffset, int* listIndex);
    RESULT_TYPE* waitForResults();
    std::vector<std::vector<DATA_TYPE>> waitForListResults();
    void wait();

    float getUtilization() const;
    float getLastRunTime() const { return m_lastRunTime; }
    float getActiveTime() const { return m_activeTime; }
    float getIdleTime() const { return m_idleTime; }
    float getTotalTime() const { return m_idleTime + m_activeTime; }

    ComputeThreadID getId() const { return m_id; }
    const std::string& getName() const { return m_name; }
    size_t getTotalCalculatedElements() const { return m_totalCalculatedElements; }
    size_t getLastCalculatedElements() const { return m_lastCalculatedElements; }
    WorkerThreadType getType() const { return m_type; }
    size_t getGpuMaxBatchSize() const {
        if(m_type != GPU)
            throw std::runtime_error("GpuMaxBatchSize can only be retrieved for GPU-type compute threads");

        return m_gpuRuntime.maxGpuBatchSize;
    }

private:
    void initDevices();
    void prepareForElements(size_t numOfElements);
    void doWorkCpu(const AssignedWork& work, RESULT_TYPE* results, int* listIndex);
    void doWorkGpu(const AssignedWork& work, RESULT_TYPE* results, int* listIndex);
    void finalize();

    void start(WorkDispatcher workDispatcher, RESULT_TYPE* localResults, size_t indexOffset, int* listIndex);

    void log(const char* text, ...);

    ComputeThreadID m_id;
    std::string m_name;
    WorkerThreadType m_type;
    const DesConfig& m_config;
    const std::vector<unsigned long long> m_indexSteps;
    int m_rank;
    CpuConfig m_cpuConfig;
    GpuConfig m_gpuConfig;

    Stopwatch m_idleStopwatch;
    std::thread m_thread;

    size_t m_totalCalculatedElements;
    size_t m_lastCalculatedElements;

    float m_idleTime;
    float m_activeTime;
    float m_lastRunTime;
    float m_lastIdleTime;

    struct {
        bool initialized = false;
        bool available = false;
        nvmlDevice_t gpuHandle;
        std::vector<nvmlSample_t> samples;
        unsigned long long lastSeenTimeStamp = 0;

        float totalUtilization = 0;
        unsigned long numOfSamples = 0;
    } m_nvml;

    struct {
        RESULT_TYPE* deviceResults = nullptr;   // GPU Memory for results
        int* deviceListIndexPtr = nullptr;      // GPU Memory for list index for synchronization when saving the results as a list of points
        void* deviceDataPtr = nullptr;          // GPU Memory to store the model's constant data
        std::vector<cudaStream_t> streams;

        cudaDeviceProp deviceProp;

        unsigned long allocatedElements = 0;
        unsigned long maxGpuBatchSize;
        bool useSharedMemoryForData;
        bool useConstantMemoryForData;
        int maxSharedPoints;
        int availableSharedMemory;
    } m_gpuRuntime;

    struct {
        float startUptime = -1.f;
        float startIdleTime = -1.f;
        float averageUtilization = 0.f;
    } m_cpuRuntime;
};

}
