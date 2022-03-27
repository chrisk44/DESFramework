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

#include "defs.h"
#include "stopwatch.h"
#include "types.h"

enum WorkerThreadType {
    CPU,
    GPU
};

typedef std::function<void(RESULT_TYPE*, const Limit*, unsigned int, unsigned long, void*, int*, const unsigned long long*, unsigned long, bool, int)> CallCpuKernelCallback;
typedef std::function<void(int, int, unsigned long, cudaStream_t, RESULT_TYPE*, unsigned long, const unsigned int, const unsigned long, const unsigned long, void*, int, bool, bool, int*, const int)> CallGpuKernelCallback;

typedef int ComputeThreadID;

class ParallelFramework;

class ComputeThread {
public:
    ComputeThread(ComputeThreadID id,
                  std::string name,
                  WorkerThreadType type,
                  ParallelFramework& framework,
                  ThreadCommonData& tcd,
                  CallCpuKernelCallback callCpuKernel,
                  CallGpuKernelCallback callGpuKernel);

    // Dummy copy constructor which does not actually copy the whole object, just recreates it in the new location
    ComputeThread(const ComputeThread& other)
        : ComputeThread(other.m_id, other.m_name, other.m_type, other.m_framework, other.m_tcd, other.m_callCpuKernel, other.m_callGpuKernel)
    {}

    ~ComputeThread();

    void dispatch(size_t batchSize);
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

private:
    void init();
    void prepareForElements(size_t numOfElements);
    AssignedWork getBatch(size_t batchSize);
    void doWorkCpu(const AssignedWork& work, RESULT_TYPE* results);
    void doWorkGpu(const AssignedWork& work, RESULT_TYPE* results);
    void finalize();

    void start(size_t batchSize);

    void log(const char* text, ...);

    ComputeThreadID m_id;
    std::string m_name;
    WorkerThreadType m_type;
    ParallelFramework& m_framework;
    ThreadCommonData& m_tcd;
    CallCpuKernelCallback m_callCpuKernel;
    CallGpuKernelCallback m_callGpuKernel;
    int m_rank;

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
