#include "computeThread.h"
#include "framework.h"
#include "utilities.h"

#include <cstring>
#include <nvml.h>
#include <stdarg.h>

ComputeThread::ComputeThread(int id, std::string name, WorkerThreadType type, ParallelFramework& framework, ThreadCommonData& tcd, CallCpuKernelCallback callCpuKernel, CallGpuKernelCallback callGpuKernel)
    : m_id(id),
      m_name(std::move(name)),
      m_type(type),
      m_framework(framework),
      m_tcd(tcd),
      m_callCpuKernel(callCpuKernel),
      m_callGpuKernel(callGpuKernel),
      m_rank(m_framework.getRank()),
      m_totalCalculatedElements(0),
      m_lastCalculatedElements(0),
      m_idleTime(0.f),
      m_activeTime(0.f)
{
    init();
    m_idleStopwatch.start();
}

ComputeThread::~ComputeThread() {
    if(m_thread.joinable())
        m_thread.join();

    finalize();
}

void ComputeThread::dispatch(size_t batchSize) {
    if(m_thread.joinable()) throw std::runtime_error("Compute thread is already running or was not joined");

    m_idleStopwatch.stop();
    m_lastIdleTime = m_idleStopwatch.getMsec();
    m_idleTime += m_lastIdleTime;
    m_thread = std::thread([this, batchSize](){
        start(batchSize);
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
    static thread_local char buf[65536];

    va_list args;
    va_start(args, text);
    vsnprintf(buf, sizeof(buf), text, args);
    va_end(args);

    printf("[%d] Compute thread %d: %s", m_rank, getId(), buf);
}

void ComputeThread::init() {
    const ParallelFrameworkParameters& parameters = m_framework.getParameters();

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
                log("[E] Failed to get device handle for gpu %d: %s\n", m_id, nvmlErrorString(result));
            }
        } else {
            log("[E] Failed to initialize NVML: %s\n", nvmlErrorString(result));
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
                log("[E] Failed to get initial utilization samples for device: %s\n", nvmlErrorString(result));
                m_nvml.available = false;
            }
        }

        // Select gpu[id]
        cudaSetDevice(m_id);

        // Calculate the max batch size for the device
        m_gpuRuntime.maxGpuBatchSize = getMaxGPUBytesForGpu(m_id);
        if(parameters.resultSaveType == SAVE_TYPE_ALL)
            m_gpuRuntime.maxGpuBatchSize /= sizeof(RESULT_TYPE);
        else
            m_gpuRuntime.maxGpuBatchSize /= parameters.D * sizeof(DATA_TYPE);

        // Get device's properties for shared memory
        cudaGetDeviceProperties(&m_gpuRuntime.deviceProp, m_id);

        // Use constant memory for data if they fit
        m_gpuRuntime.useConstantMemoryForData = parameters.dataSize > 0 &&
            parameters.dataSize <= (MAX_CONSTANT_MEMORY - parameters.D * (sizeof(Limit) + sizeof(unsigned long long)));

        // Max use 1/4 of the available shared memory for data, the rest will be used for each thread to store their point (x) and index vector (i)
        // This seems to be worse than both global and constant memory
        m_gpuRuntime.useSharedMemoryForData = false && parameters.dataSize > 0 && !m_gpuRuntime.useConstantMemoryForData &&
                                 parameters.dataSize <= m_gpuRuntime.deviceProp.sharedMemPerBlock / 4;

        // How many bytes are left in shared memory after using it for the model's data
        m_gpuRuntime.availableSharedMemory = m_gpuRuntime.deviceProp.sharedMemPerBlock - (m_gpuRuntime.useSharedMemoryForData ? parameters.dataSize : 0);

        // How many points can fit in shared memory (for each point we need D*DATA_TYPEs (for x) and D*u_int (for indices))
        m_gpuRuntime.maxSharedPoints = m_gpuRuntime.availableSharedMemory / (parameters.D * (sizeof(DATA_TYPE) + sizeof(unsigned int)));

        #ifdef DBG_START_STOP
            if(parameters.printProgress){
                log("useSharedMemoryForData = %d\n", m_gpuRuntime.useSharedMemoryForData);
                log("useConstantMemoryForData = %d\n", m_gpuRuntime.useConstantMemoryForData);
                log("availableSharedMemory = %d bytes\n", m_gpuRuntime.availableSharedMemory);
                log("maxSharedPoints = %d\n", m_gpuRuntime.maxSharedPoints);
            }
        #endif

        // Create streams
        m_gpuRuntime.streams.resize(parameters.gpuStreams);
        for(int i=0; i<parameters.gpuStreams; i++){
            cudaStreamCreate(&m_gpuRuntime.streams[i]);
            cce();
        }

        // Allocate memory on device
        cudaMalloc(&m_gpuRuntime.deviceResults, m_gpuRuntime.allocatedElements * sizeof(RESULT_TYPE));	cce();
        cudaMalloc(&m_gpuRuntime.deviceListIndexPtr, sizeof(int));							cce();
        // If we have static model data but won't use constant memory, allocate global memory for it
        if(parameters.dataSize > 0 && !m_gpuRuntime.useConstantMemoryForData){
            cudaMalloc(&m_gpuRuntime.deviceDataPtr, parameters.dataSize);					cce();
        }

        #ifdef DBG_MEMORY
            log("deviceResults: 0x%x\n", (void*) m_gpuRuntime.deviceResults);
            log("deviceListIndexPtr: 0x%x\n", (void*) m_gpuRuntime.deviceListIndexPtr);
            log("deviceDataPtr: 0x%x\n", (void*) m_gpuRuntime.deviceDataPtr);
        #endif

        // Copy limits, idxSteps, and constant data to device
        #ifdef DBG_MEMORY
            log("Copying limits at constant memory with offset %d\n", 0);
            log("Copying idxSteps at constant memory with offset %lu\n", parameters.D * sizeof(Limit));
        #endif
        cudaMemcpyToSymbolWrapper(
            m_framework.getLimits().data(), parameters.D * sizeof(Limit), 0);
        cce();

        cudaMemcpyToSymbolWrapper(
            m_framework.getIndexSteps().data(), parameters.D * sizeof(unsigned long long),
            parameters.D * sizeof(Limit));
        cce();

        // If we have data for the model...
        if(parameters.dataSize > 0){
            // If we can use constant memory, copy it there
            if(m_gpuRuntime.useConstantMemoryForData){
                #ifdef DBG_MEMORY
                    log("Copying data at constant memory with offset %lu\n", parameters.D * (sizeof(Limit) + sizeof(unsigned long long)));
                #endif
                cudaMemcpyToSymbolWrapper(
                    parameters.dataPtr, parameters.dataSize,
                    parameters.D * (sizeof(Limit) + sizeof(unsigned long long)));
                cce()
            }
            // else copy the data to the global memory, either to be read from there or to be copied to shared memory
            else{
                cudaMemcpy(m_gpuRuntime.deviceDataPtr, parameters.dataPtr, parameters.dataSize, cudaMemcpyHostToDevice);
                cce();
            }
        }
    } else {
        getCpuStats(&m_cpuRuntime.startUptime, &m_cpuRuntime.startIdleTime);
    }
}

void ComputeThread::prepareForElements(size_t numOfElements) {
    // TODO: Move this to initialization and allocate as much memory as possible
    if (m_type == WorkerThreadType::GPU && m_gpuRuntime.allocatedElements < numOfElements && m_gpuRuntime.allocatedElements < m_gpuRuntime.maxGpuBatchSize) {
        #ifdef DBG_MEMORY
            size_t prevAllocatedElements = m_gpuRuntime.allocatedElements;
        #endif

        m_gpuRuntime.allocatedElements = std::min(numOfElements, m_gpuRuntime.maxGpuBatchSize);

        #ifdef DBG_MEMORY
            log("Allocating more GPU memory (%lu -> %lu elements, %lu MB)\n", prevAllocatedElements, m_gpuRuntime.allocatedElements, (m_gpuRuntime.allocatedElements*sizeof(RESULT_TYPE)) / (1024 * 1024));
            fflush(stdout);
        #endif

        // Reallocate memory on device
        cudaFree(m_gpuRuntime.deviceResults);
        cce();
        cudaMalloc(&m_gpuRuntime.deviceResults, m_gpuRuntime.allocatedElements * sizeof(RESULT_TYPE));
        cce();

        #ifdef DBG_MEMORY
            log("deviceResults = 0x%x\n", m_gpuRuntime.deviceResults);
        #endif
    }
}

AssignedWork ComputeThread::getBatch(size_t batchSize) {
    AssignedWork work;
    {
        std::lock_guard<std::mutex> lock(m_tcd.syncMutex);

        // Get the current global batch start point as our starting point
        work.startPoint = m_tcd.globalBatchStart;
        // Increment the global batch start point by our batch size
        m_tcd.globalBatchStart += batchSize;

        // Check for globalBatchStart overflow and limit it to globalLast+1 to avoid later overflows
        // If the new globalBatchStart is smaller than our local start point, the increment caused an overflow
        // If the localStart point in larger than the global last, then the elements have already been exhausted
        if(m_tcd.globalBatchStart < work.startPoint || work.startPoint > m_tcd.globalLast){
            // log("Fixing globalBatchStart from %lu to %lu\n", tcd.globalBatchStart, tcd.globalLast + 1);
            m_tcd.globalBatchStart = m_tcd.globalLast + 1;
        }
    }

    if(work.startPoint > m_tcd.globalLast){
        work.startPoint = 0;
        work.numOfElements = 0;
    } else {
        size_t last = std::min(work.startPoint + batchSize - 1 , m_tcd.globalLast);
        work.numOfElements = last - work.startPoint + 1;
    }

    return work;
}

void ComputeThread::doWorkCpu(const AssignedWork &work, RESULT_TYPE* results) {
    const auto& parameters = m_framework.getParameters();
    m_callCpuKernel(results, m_framework.getLimits().data(), parameters.D, work.numOfElements, parameters.dataPtr, parameters.resultSaveType == SAVE_TYPE_ALL ? nullptr : &m_tcd.listIndex,
                    m_framework.getIndexSteps().data(), work.startPoint, parameters.cpuDynamicScheduling, parameters.cpuComputeBatchSize);
}

void ComputeThread::doWorkGpu(const AssignedWork &work, RESULT_TYPE* results) {
    const auto& parameters = m_framework.getParameters();

    // Initialize the list index counter
    cudaMemset(m_gpuRuntime.deviceListIndexPtr, 0, sizeof(int));

    // Divide the chunk to smaller chunks to scatter accross streams
    unsigned long elementsPerStream = work.numOfElements / parameters.gpuStreams;
    bool onlyOne = false;
    unsigned long skip = 0;
    if(elementsPerStream == 0){
        elementsPerStream = work.numOfElements;
        onlyOne = true;
    }

    // Queue the chunks to the streams
    for(int i=0; i<parameters.gpuStreams; i++){
        // Adjust elementsPerStream for last stream (= total-queued)
        if(i == parameters.gpuStreams - 1){
            elementsPerStream = work.numOfElements - skip;
        }else{
            elementsPerStream = std::min(elementsPerStream, work.numOfElements - skip);
        }

        // Queue the kernel in stream[i] (each GPU thread gets COMPUTE_BATCH_SIZE elements to calculate)
        int gpuThreads = (elementsPerStream + parameters.computeBatchSize - 1) / parameters.computeBatchSize;

        // Minimum of (minimum of user-defined block size and number of threads to go to this stream) and number of points that can fit in shared memory
        int blockSize = std::min(std::min(parameters.blockSize, gpuThreads), m_gpuRuntime.maxSharedPoints);
        int numOfBlocks = (gpuThreads + blockSize - 1) / blockSize;

        #ifdef DBG_QUEUE
            log("Queueing %lu elements in stream %d (%d gpuThreads, %d blocks, %d block size), with skip=%lu\n", elementsPerStream, i, gpuThreads, numOfBlocks, blockSize, skip);
        #endif

        // Note: Point at the start of deviceResults, because the offset (because of computeBatchSize) is calculated in the kernel
        m_callGpuKernel(numOfBlocks, blockSize, m_gpuRuntime.deviceProp.sharedMemPerBlock, m_gpuRuntime.streams[i],
            m_gpuRuntime.deviceResults, work.startPoint,
            parameters.D, elementsPerStream, skip, m_gpuRuntime.deviceDataPtr,
            parameters.dataSize, m_gpuRuntime.useSharedMemoryForData, m_gpuRuntime.useConstantMemoryForData,
            parameters.resultSaveType == SAVE_TYPE_ALL ? nullptr : m_gpuRuntime.deviceListIndexPtr,
            parameters.computeBatchSize
        );

        // Queue the memcpy in stream[i] only if we are saving as SAVE_TYPE_ALL (otherwise the results will be fetched at the end of the current computation)
        if(parameters.resultSaveType == SAVE_TYPE_ALL){
            cudaMemcpyAsync(&results[skip], &m_gpuRuntime.deviceResults[skip], elementsPerStream*sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost, m_gpuRuntime.streams[i]);
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

    // If we are saving as SAVE_TYPE_LIST, fetch the results
    if(parameters.resultSaveType == SAVE_TYPE_LIST){
        int gpuListIndex, globalListIndexOld;
        // Get the current list index from the GPU
        cudaMemcpy(&gpuListIndex, m_gpuRuntime.deviceListIndexPtr, sizeof(int), cudaMemcpyDeviceToHost);

        // Increment the global list index counter
        globalListIndexOld = __sync_fetch_and_add(&m_tcd.listIndex, gpuListIndex);

        // Get the results from the GPU
        cudaMemcpy(&((DATA_TYPE*)results)[globalListIndexOld], m_gpuRuntime.deviceResults, gpuListIndex * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    }
}

void ComputeThread::start(size_t batchSize){

    const auto& parameters = m_framework.getParameters();

    Stopwatch activeStopwatch;
    activeStopwatch.start();

    #ifdef DBG_TIME
        Stopwatch sw;
        float time_assign=0, time_allocation=0, time_calc=0;
        sw.start();
    #endif

    #ifdef DBG_START_STOP
        log("Woke up\n");
    #endif

    size_t numOfCalculatedElements = 0;
    while(1){
        #ifdef DBG_TIME
            sw.start();
        #endif

        Stopwatch syncStopwatch;
        syncStopwatch.start();

        AssignedWork work = getBatch(batchSize);

        syncStopwatch.stop();
        m_idleTime += syncStopwatch.getMsec();

        if(work.numOfElements == 0)
            break;

        RESULT_TYPE* localResults;
        if(parameters.resultSaveType == SAVE_TYPE_LIST)
            localResults = m_tcd.results;
        else
            localResults = &m_tcd.results[work.startPoint - m_tcd.globalFirst];

        #ifdef DBG_TIME
            sw.stop();
            time_assign += sw.getMsec();
        #endif

        #ifdef DBG_DATA
            log("Got %lu elements starting from %lu\n", work.numOfElements, work.startPoint);
            fflush(stdout);
        #else
            #ifdef DBG_START_STOP
                log("Running for %lu elements...\n", work.numOfElements);
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
            sw.start();
        #endif

        /*****************************************************************
        ******************** Calculate the results ***********************
        ******************************************************************/
        if (m_type == WorkerThreadType::GPU) {
            doWorkGpu(work, localResults);
        } else {
            doWorkCpu(work, localResults);
        }

        numOfCalculatedElements += work.numOfElements;

        #ifdef DBG_TIME
            sw.stop();
            time_calc += sw.getMsec();
        #endif

        #ifdef DBG_RESULTS
            if(parameters.resultSaveType == SAVE_TYPE_ALL){
                log("Results are: ");
                for (unsigned long i = 0; i < work.numOfElements; i++) {
                    printf("%f ", ((DATA_TYPE *)localResults)[i]);
                }
                printf("\n");
            }
        #endif

        #ifdef DBG_START_STOP
            log("Finished calculation\n");
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
                log("[E1] Failed to get utilization samples for device: %s\n", nvmlErrorString(result));
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
                    log("[E2] Failed to get utilization samples for device: %s\n", nvmlErrorString(result));
                }
            }
        }
        }

    #ifdef DBG_TIME
        log("Benchmark:\n");
        log("Time for assignments: %f ms\n", time_assign);
        log("Time for allocations: %f ms\n", time_allocation);
        log("Time for calcs: %f ms\n", time_calc);
    #endif

    #ifdef DBG_START_STOP
        log("Finished job\n");
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
                log("[E] Failed to shutdown NVML: %s\n", nvmlErrorString(result));
        }
    }
}

