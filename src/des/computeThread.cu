#include "computeThread.h"
#include "desf.h"
#include "utilities.h"
#include "gpuKernel.h"

#include <cstring>
#include <nvml.h>
#include <stdarg.h>

namespace desf {

ComputeThread::ComputeThread(int id, std::string name, WorkerThreadType type, const DesConfig& config, const std::vector<unsigned long long>& indexSteps)
    : m_id(id),
      m_name(std::move(name)),
      m_type(type),
      m_config(config),
      m_indexSteps(indexSteps),
      m_totalCalculatedElements(0),
      m_lastCalculatedElements(0),
      m_idleTime(0.f),
      m_activeTime(0.f)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    if(type == CPU) {
        m_cpuConfig = m_config.cpu;
    } else {
        auto it = m_config.gpu.find(m_id);
        if(it == m_config.gpu.end())
            throw std::invalid_argument("No GPU config found for gpu " +std::to_string(m_id));

        m_gpuConfig = it->second;
    }

    initDevices();

    m_idleStopwatch.start();
}

ComputeThread::~ComputeThread() {
    if(m_thread.joinable())
        m_thread.join();

    finalize();
}

void ComputeThread::dispatch(ComputeEnvironment& computeEnvironment) {
    if(m_thread.joinable()) throw std::runtime_error("Compute thread is already running or was not joined");

    m_idleStopwatch.stop();
    m_lastIdleTime = m_idleStopwatch.getMsec();
    m_idleTime += m_lastIdleTime;
    m_thread = std::thread([this, &computeEnvironment](){
        start(computeEnvironment);
        m_idleStopwatch.start();
    });
}

void ComputeThread::wait() {
    if(!m_thread.joinable()) throw std::runtime_error("Compute thread is not running");

    m_thread.join();
}

float ComputeThread::getUtilization() const {
    if(m_type == WorkerThreadType::GPU){
        if(m_nvml.available && m_nvml.numOfSamples > 0){
            return m_nvml.totalUtilization / m_nvml.numOfSamples;
        } else {
            return 0.f;
        }
    } else {
        return m_cpuRuntime.averageUtilization;
    }
}

void ComputeThread::log(const char *text, ...) {
    static thread_local char buf[LOG_BUFFER_SIZE];

    va_list args;
    va_start(args, text);
    vsnprintf(buf, sizeof(buf), text, args);
    va_end(args);

    printf("%s [%d]       Compute thread %d: %s\n", getTimeString().c_str(), m_rank, getId(), buf);
}

void ComputeThread::initDevices() {
    if (m_type == WorkerThreadType::GPU) {
        // Initialize NVML to monitor the GPU
        m_nvml.available = false;
        nvmlReturn_t result = nvmlInit();
        if (result == NVML_SUCCESS){
            m_nvml.initialized = true;

            result = nvmlDeviceGetHandleByIndex(m_id, &m_nvml.gpuHandle);
            if(result == NVML_SUCCESS){
                m_nvml.available = true;
            }else{
                log("[E] Failed to get device handle for gpu %d: %s", m_id, nvmlErrorString(result));
            }
        } else {
            log("[E] Failed to initialize NVML: %s", nvmlErrorString(result));
            m_nvml.available = false;
        }

        if(m_nvml.available){
            // Get the device's name for later
            char gpuName[NVML_DEVICE_NAME_BUFFER_SIZE];
            nvmlDeviceGetName(m_nvml.gpuHandle, gpuName, NVML_DEVICE_NAME_BUFFER_SIZE);
            m_name = gpuName;

            // Get samples to save the current timestamp
            unsigned int temp = 1;
            m_nvml.samples.resize(1);
            nvmlValueType_t sampleValType;
            // Read them one by one to avoid allocating memory for the whole buffer
            while((result = nvmlDeviceGetSamples(m_nvml.gpuHandle, NVML_GPU_UTILIZATION_SAMPLES, m_nvml.lastSeenTimeStamp, &sampleValType, &temp, m_nvml.samples.data())) == NVML_SUCCESS && temp > 0){
                m_nvml.lastSeenTimeStamp = m_nvml.samples[temp-1].timeStamp;
            }

            if (result != NVML_SUCCESS && result != NVML_ERROR_NOT_FOUND) {
                log("[E] Failed to get initial utilization samples for device: %s", nvmlErrorString(result));
                m_nvml.available = false;
            }
        }

        // Select gpu[id]
        cudaSetDevice(m_id);

        m_gpuRuntime.maxBytes = getMaxGPUBytesForGpu(m_id);
        if(m_gpuRuntime.maxBytes == 0)
            throw std::runtime_error("No memory available on GPU " + std::to_string(m_id));

        #ifdef DBG_MEMORY
            log("Max memory is %lu bytes", m_gpuRuntime.maxBytes);
        #endif

        // Calculate the max batch size for the device
        if(m_config.output.overrideMemoryRestrictions) {
            m_gpuRuntime.maxGpuBatchSize = std::numeric_limits<size_t>::max();
        } else {
            m_gpuRuntime.maxGpuBatchSize = m_gpuRuntime.maxBytes;
            if(m_config.resultSaveType == SAVE_TYPE_ALL)
                m_gpuRuntime.maxGpuBatchSize /= sizeof(RESULT_TYPE);
            else
                m_gpuRuntime.maxGpuBatchSize /= sizeof(DATA_TYPE);
        }

        // Get device's properties for shared memory
        cudaGetDeviceProperties(&m_gpuRuntime.deviceProp, m_id);

        // Use constant memory for data if they fit
        m_gpuRuntime.useConstantMemoryForData = m_config.model.dataSize > 0 &&
            m_config.model.dataSize <= (MAX_CONSTANT_MEMORY - m_config.model.D * (sizeof(Limit) + sizeof(unsigned long long)));

        // Max use 1/4 of the available shared memory for data, the rest will be used for each thread to store their point (x) and index vector (i)
        // This seems to be worse than both global and constant memory
        m_gpuRuntime.useSharedMemoryForData = false && m_config.model.dataSize > 0 && !m_gpuRuntime.useConstantMemoryForData &&
                                 m_config.model.dataSize <= m_gpuRuntime.deviceProp.sharedMemPerBlock / 4;

        // How many bytes are left in shared memory after using it for the model's data
        m_gpuRuntime.availableSharedMemory = m_gpuRuntime.deviceProp.sharedMemPerBlock - (m_gpuRuntime.useSharedMemoryForData ? m_config.model.dataSize : 0);

        // How many points can fit in shared memory (for each point we need D*DATA_TYPEs (for x) and D*u_int (for indices))
        m_gpuRuntime.maxSharedPoints = m_gpuRuntime.availableSharedMemory / (m_config.model.D * (sizeof(DATA_TYPE) + sizeof(unsigned int)));

        #ifdef DBG_START_STOP
            if(m_config.printProgress){
                log("useSharedMemoryForData = %d", m_gpuRuntime.useSharedMemoryForData);
                log("useConstantMemoryForData = %d", m_gpuRuntime.useConstantMemoryForData);
                log("availableSharedMemory = %d bytes", m_gpuRuntime.availableSharedMemory);
                log("maxSharedPoints = %d", m_gpuRuntime.maxSharedPoints);
            }
        #endif

        // Create streams
        m_gpuRuntime.streams.resize(m_gpuConfig.streams);
        for(int i=0; i<m_gpuConfig.streams; i++){
            cudaStreamCreate(&m_gpuRuntime.streams[i]);
            cce();
        }

        // Allocate memory on device
        cudaMalloc(&m_gpuRuntime.deviceResults, m_gpuRuntime.allocatedBytes); cce();
        cudaMalloc(&m_gpuRuntime.deviceListIndexPtr, sizeof(int));							cce();
        // If we have static model data but won't use constant memory, allocate global memory for it
        if(m_config.model.dataSize > 0 && !m_gpuRuntime.useConstantMemoryForData){
            cudaMalloc(&m_gpuRuntime.deviceDataPtr, m_config.model.dataSize);					cce();
        }

        #ifdef DBG_MEMORY
            log("deviceResults: %p", (void*) m_gpuRuntime.deviceResults);
            log("deviceListIndexPtr: %p", (void*) m_gpuRuntime.deviceListIndexPtr);
            log("deviceDataPtr: %p", (void*) m_gpuRuntime.deviceDataPtr);
        #endif

        // Copy limits, idxSteps, and constant data to device
        #ifdef DBG_MEMORY
            log("Copying limits at constant memory with offset %d", 0);
            log("Copying idxSteps at constant memory with offset %lu", m_config.model.D * sizeof(Limit));
        #endif
        cudaMemcpyToSymbolWrapper(
            m_config.limits.data(), m_config.model.D * sizeof(Limit), 0);
        cce();

        cudaMemcpyToSymbolWrapper(
            m_indexSteps.data(), m_config.model.D * sizeof(unsigned long long),
            m_config.model.D * sizeof(Limit));
        cce();

        // If we have data for the model...
        if(m_config.model.dataSize > 0){
            // If we can use constant memory, copy it there
            if(m_gpuRuntime.useConstantMemoryForData){
                #ifdef DBG_MEMORY
                    log("Copying data at constant memory with offset %lu", m_config.model.D * (sizeof(Limit) + sizeof(unsigned long long)));
                #endif
                cudaMemcpyToSymbolWrapper(
                    m_config.model.dataPtr, m_config.model.dataSize,
                    m_config.model.D * (sizeof(Limit) + sizeof(unsigned long long)));
                cce()
            }
            // else copy the data to the global memory, either to be read from there or to be copied to shared memory
            else{
                cudaMemcpy(m_gpuRuntime.deviceDataPtr, m_config.model.dataPtr, m_config.model.dataSize, cudaMemcpyHostToDevice);
                cce();
            }
        }
    } else {
        getCpuStats(&m_cpuRuntime.startUptime, &m_cpuRuntime.startIdleTime);
    }
}

void ComputeThread::prepareForElements(size_t numOfElements) {
    if(m_type != WorkerThreadType::GPU)
        return;

    size_t toAllocateBytes;
    if(m_config.output.overrideMemoryRestrictions) {
        toAllocateBytes = m_gpuRuntime.maxBytes;
    } else if(m_config.resultSaveType == SAVE_TYPE_ALL) {
        toAllocateBytes = numOfElements * sizeof(RESULT_TYPE);
    } else {
        toAllocateBytes = numOfElements * sizeof(DATA_TYPE);
    }

    if (m_gpuRuntime.allocatedBytes < toAllocateBytes) {
        #ifdef DBG_MEMORY
            log("Allocating more GPU memory (%lu -> %lu MB)", m_gpuRuntime.allocatedBytes/(1024*1024), toAllocateBytes/(1024*1024));
            fflush(stdout);
        #endif

        // Reallocate memory on device
        cudaFree(m_gpuRuntime.deviceResults);
        cce();
        cudaMalloc(&m_gpuRuntime.deviceResults, toAllocateBytes);
        cce();

        m_gpuRuntime.allocatedBytes = toAllocateBytes;

        #ifdef DBG_MEMORY
            log("deviceResults = %p", m_gpuRuntime.deviceResults);
        #endif
    }
}

void ComputeThread::doWorkCpu(const AssignedWork &work, ComputeEnvironment& env, float* t_calc, float* t_memcpy) {
    Stopwatch sw;
    sw.start();
//    cpu_kernel(m_config.cpu.forwardModel,
//               m_config.cpu.objective,
//               m_config.model.D,
//               m_config.limits.data(),
//               m_indexSteps.data(),
//               work.startPoint,
//               work.numOfElements,
//               env.getSaveType() == SAVE_TYPE_ALL ? env.getAddrForIndex(work.startPoint) : nullptr,
//               [&env](size_t index){ env.addResult(index); },
//               m_config.model.dataPtr,
//               m_config.cpu.dynamicScheduling,
//               m_config.cpu.computeBatchSize);

    usleep(m_rank * work.numOfElements);
    sw.stop();
    if(t_calc) *t_calc = sw.getMsec();
    if(t_memcpy) *t_memcpy = 0;
}

void ComputeThread::doWorkGpu(const AssignedWork &work, ComputeEnvironment& env, float* t_calc, float* t_memcpy) {
    if(t_memcpy) *t_memcpy = 0;

    Stopwatch sw;

    sw.start();
    // Initialize the list index counter
    cudaMemset(m_gpuRuntime.deviceListIndexPtr, 0, sizeof(int));
    sw.stop();
    if(t_memcpy) *t_memcpy += sw.getMsec();

    sw.start();

    // Divide the chunk to smaller chunks to scatter accross streams
    unsigned long elementsPerStream = work.numOfElements / m_gpuConfig.streams;
    bool onlyOne = false;
    unsigned long skip = 0;
    if(elementsPerStream == 0){
        elementsPerStream = work.numOfElements;
        onlyOne = true;
    }

    RESULT_TYPE* allResults = env.getSaveType() == SAVE_TYPE_ALL ? env.getAddrForIndex(work.startPoint) : nullptr;

    // Queue the chunks to the streams
    for(int i=0; i<m_gpuConfig.streams; i++){
        // Adjust elementsPerStream for last stream (= total-queued)
        if(i == m_gpuConfig.streams - 1){
            elementsPerStream = work.numOfElements - skip;
        }else{
            elementsPerStream = std::min(elementsPerStream, work.numOfElements - skip);
        }

        // Queue the kernel in stream[i] (each GPU thread gets COMPUTE_BATCH_SIZE elements to calculate)
        int gpuThreads = (elementsPerStream + m_gpuConfig.computeBatchSize - 1) / m_gpuConfig.computeBatchSize;

        // Minimum of (minimum of user-defined block size and number of threads to go to this stream) and number of points that can fit in shared memory
        int blockSize = std::min(std::min(m_gpuConfig.blockSize, gpuThreads), m_gpuRuntime.maxSharedPoints);
        int numOfBlocks = (gpuThreads + blockSize - 1) / blockSize;

        #ifdef DBG_QUEUE
            log("Queueing %lu elements in stream %d (%d gpuThreads, %d blocks, %d block size), with skip=%lu", elementsPerStream, i, gpuThreads, numOfBlocks, blockSize, skip);
        #endif

        // Note: Point at the start of deviceResults, because the offset (because of computeBatchSize) is calculated in the kernel
        validate_kernel<<<numOfBlocks, blockSize, m_gpuRuntime.deviceProp.sharedMemPerBlock, m_gpuRuntime.streams[i]>>>(
            m_gpuConfig.forwardModel, m_gpuConfig.objective,
            m_gpuRuntime.deviceResults, work.startPoint,
            m_config.model.D, elementsPerStream, skip, m_gpuRuntime.deviceDataPtr,
            m_config.model.dataSize, m_gpuRuntime.useSharedMemoryForData, m_gpuRuntime.useConstantMemoryForData,
            m_config.resultSaveType == SAVE_TYPE_ALL ? nullptr : m_gpuRuntime.deviceListIndexPtr,
            m_gpuConfig.computeBatchSize
        );

        // Queue the memcpy in stream[i] only if we are saving as SAVE_TYPE_ALL (otherwise the results will be fetched at the end of the current computation)
        if(m_config.resultSaveType == SAVE_TYPE_ALL){
            cudaMemcpyAsync(&allResults[skip], &((RESULT_TYPE*) m_gpuRuntime.deviceResults)[skip], elementsPerStream*sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost, m_gpuRuntime.streams[i]);
        }

        // Increase skip
        skip += elementsPerStream;

        if(onlyOne)
            break;
    }

    // Wait for all streams to finish
    for(auto& stream : m_gpuRuntime.streams){
        cudaStreamSynchronize(stream);
        cce();
    }

    sw.stop();
    if(t_calc) *t_calc = sw.getMsec();

    // If we are saving as SAVE_TYPE_LIST, fetch the results
    if(m_config.resultSaveType == SAVE_TYPE_LIST){
        sw.start();

        int gpuListIndex;
        // Get the current list index from the GPU
        cudaMemcpy(&gpuListIndex, m_gpuRuntime.deviceListIndexPtr, sizeof(int), cudaMemcpyDeviceToHost); cce();

        // Get address to save the results to
        if(gpuListIndex > 0) {
            size_t* dest = env.getAddrToAddIndices(gpuListIndex);

            // Get the results from the GPU
            cudaMemcpy(dest, m_gpuRuntime.deviceResults, gpuListIndex * sizeof(size_t), cudaMemcpyDeviceToHost); cce();
        }

        sw.stop();
        if(t_memcpy) *t_memcpy += sw.getMsec();
    }
}

void ComputeThread::start(ComputeEnvironment& env)
{
    Stopwatch activeStopwatch;
    activeStopwatch.start();

    #ifdef DBG_TIME
        Stopwatch sw;
        float time_assign=0, time_allocation=0, time_calc=0, time_memcpy=0;
        sw.start();
    #endif

    #ifdef DBG_START_STOP
        log("Woke up");
    #endif

    size_t numOfCalculatedElements = 0;
    while(1){
        #ifdef DBG_TIME
            sw.start();
        #endif

        Stopwatch syncStopwatch;
        syncStopwatch.start();

        AssignedWork work = env.getWork(getId());

        syncStopwatch.stop();
        m_idleTime += syncStopwatch.getMsec();

        #ifdef DBG_TIME
            sw.stop();
            time_assign += sw.getMsec();
        #endif

        if(work.numOfElements == 0)
            break;

        #ifdef DBG_DATA
            log("Got %lu elements starting from %lu", work.numOfElements, work.startPoint);
            fflush(stdout);
        #else
            #ifdef DBG_START_STOP
                log("Running for %lu elements...", work.numOfElements);
                fflush(stdout);
            #endif
        #endif

        #ifdef DBG_TIME
            sw.start();
        #endif

        prepareForElements(work.numOfElements);

        #ifdef DBG_TIME
            sw.stop();
            time_allocation += sw.getMsec();
        #endif

        /*****************************************************************
        ******************** Calculate the results ***********************
        ******************************************************************/
        float *t_calc = nullptr;
        float *t_memcpy = nullptr;
        #ifdef DBG_TIME
            t_calc = &time_calc;
            t_memcpy = &time_memcpy;
        #endif
        if (m_type == WorkerThreadType::GPU) {
            doWorkGpu(work, env, t_calc, t_memcpy);
        } else {
            doWorkCpu(work, env, t_calc, t_memcpy);
        }

        numOfCalculatedElements += work.numOfElements;

        #ifdef DBG_RESULTS_RAW
            size_t listCount;
            size_t* listResults = m_config.resultSaveType == SAVE_TYPE_LIST ? env.getListResults(&listCount) : nullptr;

            std::string str;
            str += "[ ";
            if(m_config.resultSaveType == SAVE_TYPE_ALL){
                for(unsigned long i=0; i<work.numOfElements; i++){
                    char tmp[64];
                    sprintf(tmp, "%f ", *env.getAddrForIndex(work.startPoint + 1));
                    str += tmp;
                }
            }else{
                for(unsigned long i=0; i<listCount; i++){
                    char tmp[64];
                    sprintf(tmp, "%lu ", listResults[i]);
                    str += tmp;
                }
            }
            str += "]";
            log("Results: %s", str.c_str());
        #endif

        #ifdef DBG_START_STOP
            log("Finished calculation");
        #endif
    }

    // Stop the stopwatch
    activeStopwatch.stop();
    m_activeTime += activeStopwatch.getMsec();
    m_lastRunTime = activeStopwatch.getMsec();

    m_totalCalculatedElements += numOfCalculatedElements;
    m_lastCalculatedElements = numOfCalculatedElements;

    if(m_type == WorkerThreadType::CPU){
        float endUptime, endIdleTime;
        if(m_cpuRuntime.startUptime > 0 && m_cpuRuntime.startIdleTime > 0 && getCpuStats(&endUptime, &endIdleTime) == 0){
            m_cpuRuntime.averageUtilization = 100 - 100 * (endIdleTime - m_cpuRuntime.startIdleTime) / (endUptime - m_cpuRuntime.startUptime);

        }
    } else {
            nvmlValueType_t sampleValType;
            if(m_nvml.available){            // Get number of available samples

            unsigned int tmpSamples;
            nvmlReturn_t result = nvmlDeviceGetSamples(m_nvml.gpuHandle, NVML_GPU_UTILIZATION_SAMPLES, m_nvml.lastSeenTimeStamp, &sampleValType, &tmpSamples, NULL);
            if (result != NVML_SUCCESS && result != NVML_ERROR_NOT_FOUND) {
                log("[E1] Failed to get utilization samples for device: %s", nvmlErrorString(result));
            }else if(result == NVML_SUCCESS){

                // Make sure we have enough allocated memory for the new samples
                if(tmpSamples > m_nvml.samples.size()){
                    m_nvml.samples.resize(tmpSamples);
                }

                result = nvmlDeviceGetSamples(m_nvml.gpuHandle, NVML_GPU_UTILIZATION_SAMPLES, m_nvml.lastSeenTimeStamp, &sampleValType, &tmpSamples, m_nvml.samples.data());
                if (result == NVML_SUCCESS) {
                    m_nvml.numOfSamples += tmpSamples;
                    for(unsigned int i=0; i<tmpSamples; i++){
                        m_nvml.totalUtilization += m_nvml.samples[i].sampleValue.uiVal;
                    }
                }else if(result != NVML_ERROR_NOT_FOUND){
                    log("[E2] Failed to get utilization samples for device: %s", nvmlErrorString(result));
                }
            }
        }
        }

    #ifdef DBG_TIME
        log("Benchmark:");
        log("Time for assignments: %f ms", time_assign);
        log("Time for allocations: %f ms", time_allocation);
        log("Time for calcs: %f ms", time_calc);
        log("Time for memcpy: %f ms", time_memcpy);
    #endif

    #ifdef DBG_START_STOP
        log("Finished job");
    #endif
}

void ComputeThread::finalize() {
    if(m_type == WorkerThreadType::GPU) {
        // Make sure streams are finished and destroy them
        for(auto& stream : m_gpuRuntime.streams){
            cudaStreamDestroy(stream);
            cce();
        }

        // Deallocate device's memory
        cudaFree(m_gpuRuntime.deviceResults);			cce();
        cudaFree(m_gpuRuntime.deviceListIndexPtr);		cce();
        cudaFree(m_gpuRuntime.deviceDataPtr);			cce();

        if(m_nvml.initialized){
            nvmlReturn_t result = nvmlShutdown();
            if (result != NVML_SUCCESS)
                log("[E] Failed to shutdown NVML: %s", nvmlErrorString(result));
        }
    }
}

}
