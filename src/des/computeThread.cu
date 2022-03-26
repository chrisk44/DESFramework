#include "framework.h"

#include <cstring>

void ParallelFramework::computeThreadImpl(ComputeThreadInfo& cti, ThreadCommonData& tcd, CallCpuKernelCallback callCpuKernel, CallGpuKernelCallback callGpuKernel){
    int gpuListIndex, globalListIndexOld;
    RESULT_TYPE* localResults;
    Stopwatch idleStopwatch;

    // GPU Memory
    RESULT_TYPE* deviceResults;					// GPU Memory for results
    int* deviceListIndexPtr;					// GPU Memory for list index for synchronization when saving the results as a list of points
    void* deviceDataPtr;						// GPU Memory to store the model's constant data

    // GPU Runtime
    cudaStream_t streams[parameters.gpuStreams];
    unsigned long allocatedElements = 0;
    cudaDeviceProp deviceProp;
    unsigned long maxGpuBatchSize;
    bool useSharedMemoryForData;
    bool useConstantMemoryForData;
    int maxSharedPoints;
    int availableSharedMemory;

    // NVML
    bool nvmlInitialized = false;
    bool nvmlAvailable = false;
    nvmlReturn_t result;
    nvmlDevice_t gpuHandle;

    float totalUtilization = 0;
    unsigned long numOfSamples = 0;

    nvmlSample_t *samples = new nvmlSample_t[1];
    unsigned long numOfAllocatedSamples = 1;
    unsigned long long lastSeenTimeStamp = 0;
    nvmlValueType_t sampleValType;

    // CPU usage
    float startUptime = -1, startIdleTime = -1;
    float endUptime, endIdleTime;

    /*******************************************************************
    ************************* Initialization ***************************
    ********************************************************************/
    // Initialize device
    if (cti.id > -1) {
        // Initialize NVML to monitor the GPU
        nvmlAvailable = false;
        result = nvmlInit();
        if (result == NVML_SUCCESS){
            nvmlInitialized = true;

            result = nvmlDeviceGetHandleByIndex(cti.id, &gpuHandle);
            if(result == NVML_SUCCESS){
                nvmlAvailable = true;
            }else{
                printf("[%d] [E] Failed to get device handle for gpu %d: %s\n", rank, cti.id, nvmlErrorString(result));
            }
        } else {
            printf("[%d] [E] Failed to initialize NVML: %s\n", rank, nvmlErrorString(result));
            nvmlAvailable = false;
        }

        if(nvmlAvailable){
            // Get the device's name for later
            char gpuName[NVML_DEVICE_NAME_BUFFER_SIZE];
            nvmlDeviceGetName(gpuHandle, gpuName, NVML_DEVICE_NAME_BUFFER_SIZE);
            cti.name = gpuName;

            // Get samples to save the current timestamp
            unsigned int temp = 1;
            // Read them one by one to avoid allocating memory for the whole buffer
            while((result = nvmlDeviceGetSamples(gpuHandle, NVML_GPU_UTILIZATION_SAMPLES, lastSeenTimeStamp, &sampleValType, &temp, samples)) == NVML_SUCCESS && temp > 0){
                lastSeenTimeStamp = samples[temp-1].timeStamp;
            }

            if (result != NVML_SUCCESS && result != NVML_ERROR_NOT_FOUND) {
                printf("[%d] [E] Failed to get initial utilization samples for device: %s\n", rank, nvmlErrorString(result));
                nvmlAvailable = false;
            }
        }

        // Select gpu[id]
        cudaSetDevice(cti.id);

        // Calculate the max batch size for the device
        maxGpuBatchSize = getMaxGPUBytesForGpu(cti.id);
        if(parameters.resultSaveType == SAVE_TYPE_ALL)
            maxGpuBatchSize /= sizeof(RESULT_TYPE);
        else
            maxGpuBatchSize /= parameters.D * sizeof(DATA_TYPE);

        // Get device's properties for shared memory
        cudaGetDeviceProperties(&deviceProp, cti.id);

        // Use constant memory for data if they fit
        useConstantMemoryForData = parameters.dataSize > 0 &&
            parameters.dataSize <= (MAX_CONSTANT_MEMORY - parameters.D * (sizeof(Limit) + sizeof(unsigned long long)));

        // Max use 1/4 of the available shared memory for data, the rest will be used for each thread to store their point (x) and index vector (i)
        // This seems to be worse than both global and constant memory
        useSharedMemoryForData = false && parameters.dataSize > 0 && !useConstantMemoryForData &&
                                 parameters.dataSize <= deviceProp.sharedMemPerBlock / 4;

        // How many bytes are left in shared memory after using it for the model's data
        availableSharedMemory = deviceProp.sharedMemPerBlock - (useSharedMemoryForData ? parameters.dataSize : 0);

        // How many points can fit in shared memory (for each point we need D*DATA_TYPEs (for x) and D*u_int (for indices))
        maxSharedPoints = availableSharedMemory / (parameters.D * (sizeof(DATA_TYPE) + sizeof(unsigned int)));

        #ifdef DBG_START_STOP
            if(parameters.printProgress){
                printf("[%d] ComputeThread %d: useSharedMemoryForData = %d\n", rank, cti.id, useSharedMemoryForData);
                printf("[%d] ComputeThread %d: useConstantMemoryForData = %d\n", rank, cti.id, useConstantMemoryForData);
                printf("[%d] ComputeThread %d: availableSharedMemory = %d bytes\n", rank, cti.id, availableSharedMemory);
                printf("[%d] ComputeThread %d: maxSharedPoints = %d\n", rank, cti.id, maxSharedPoints);
            }
        #endif

        // Create streams
        for(int i=0; i<parameters.gpuStreams; i++){
            cudaStreamCreate(&streams[i]);
            cce();
        }

        // Allocate memory on device
        cudaMalloc(&deviceResults, allocatedElements * sizeof(RESULT_TYPE));	cce();
        cudaMalloc(&deviceListIndexPtr, sizeof(int));							cce();
        // If we have static model data but won't use constant memory, allocate global memory for it
        if(parameters.dataSize > 0 && !useConstantMemoryForData){
            cudaMalloc(&deviceDataPtr, parameters.dataSize);					cce();
        }

        #ifdef DBG_MEMORY
            printf("[%d] ComputeThread %d: deviceResults: 0x%x\n", rank, cti.id, (void*) deviceResults);
            printf("[%d] ComputeThread %d: deviceListIndexPtr: 0x%x\n", rank, cti.id, (void*) deviceListIndexPtr);
            printf("[%d] ComputeThread %d: deviceDataPtr: 0x%x\n", rank, cti.id, (void*) deviceDataPtr);
        #endif

        // Copy limits, idxSteps, and constant data to device
        #ifdef DBG_MEMORY
            printf("[%d] ComputeThread %d: Copying limits at constant memory with offset %d\n",
                                            rank, cti.id, 0);
            printf("[%d] ComputeThread %d: Copying idxSteps at constant memory with offset %lu\n",
                                            rank, cti.id, parameters.D * sizeof(Limit));
        #endif
        cudaMemcpyToSymbolWrapper(
            limits.data(), parameters.D * sizeof(Limit), 0);
        cce();

        cudaMemcpyToSymbolWrapper(
            idxSteps.data(), parameters.D * sizeof(unsigned long long),
            parameters.D * sizeof(Limit));
        cce();

        // If we have data for the model...
        if(parameters.dataSize > 0){
            // If we can use constant memory, copy it there
            if(useConstantMemoryForData){
                #ifdef DBG_MEMORY
                    printf("[%d] ComputeThread %d: Copying data at constant memory with offset %lu\n",
                                        rank, cti.id, parameters.D * (sizeof(Limit) + sizeof(unsigned long long)));
                #endif
                cudaMemcpyToSymbolWrapper(
                    parameters.dataPtr, parameters.dataSize,
                    parameters.D * (sizeof(Limit) + sizeof(unsigned long long)));
                cce()
            }
            // else copy the data to the global memory, either to be read from there or to be copied to shared memory
            else{
                cudaMemcpy(deviceDataPtr, parameters.dataPtr, parameters.dataSize, cudaMemcpyHostToDevice);
                cce();
            }
        }
    } else {
        cti.name = "CPU";
        getCpuStats(&startUptime, &startIdleTime);
    }


    /*******************************************************************
    ************************* Execution loop ***************************
    ********************************************************************/
    while(true){
        #ifdef DBG_TIME
            Stopwatch sw;
            float time_sleep, time_assign=0, time_allocation=0, time_queue=0, time_calc=0;
            sw.start();
        #endif

        /*****************************************************************
        ************* Wait for data from coordinateThread ****************
        ******************************************************************/
        #ifdef DBG_START_STOP
            printf("[%d] ComputeThread %d: Ready\n", rank, cti.id);
        #endif

        idleStopwatch.start();
        cti.waitStartSemaphore();
        idleStopwatch.stop();
        cti.idleTime += idleStopwatch.getMsec();

        #ifdef DBG_TIME
            sw.stop();
            time_sleep = sw.getMsec();
        #endif

        cti.stopwatch.start();

        #ifdef DBG_START_STOP
            printf("[%d] ComputeThread %d: Waking up...\n", rank, cti.id);
        #endif

        // If coordinator thread is signaling to terminate...
        if (tcd.globalFirst > tcd.globalLast)
            break;

        /*****************************************************************
         ***************** Main batch assignment loop ********************
         ******* (will run only once if !slaveDynamicScheduling) *********
         *****************************************************************/
        while(cti.batchSize > 0){	// No point in running with batch size = 0

            unsigned long localStartPoint;
            unsigned long localLast;
            unsigned long localNumOfElements;

            /*****************************************************************
            ************************** Get a batch ***************************
            ******************************************************************/
            #ifdef DBG_TIME
                sw.start();
            #endif

            idleStopwatch.start();
            {
                std::lock_guard<std::mutex> lock(tcd.syncMutex);
                idleStopwatch.stop();
                cti.idleTime += idleStopwatch.getMsec();

                // Get the current global batch start point as our starting point
                localStartPoint = tcd.globalBatchStart;
                // Increment the global batch start point by our batch size
                tcd.globalBatchStart += cti.batchSize;

                // Check for globalBatchStart overflow and limit it to globalLast+1 to avoid later overflows
                // If the new globalBatchStart is smaller than our local start point, the increment caused an overflow
                // If the localStart point in larger than the global last, then the elements have already been exhausted
                if(tcd.globalBatchStart < localStartPoint || localStartPoint > tcd.globalLast){
                    // printf("[%d] ComputeThread %d: Fixing globalBatchStart from %lu to %lu\n",
                    // 				rank, cti.id, tcd.globalBatchStart, tcd.globalLast + 1);
                    tcd.globalBatchStart = tcd.globalLast + 1;
                }
            }

            // If we are out of elements then terminate the loop
            if(localStartPoint > tcd.globalLast)
                break;

            localLast = std::min(localStartPoint + cti.batchSize - 1 , tcd.globalLast);
            localNumOfElements = localLast - localStartPoint + 1;

            if(parameters.resultSaveType == SAVE_TYPE_LIST)
                localResults = tcd.results;
            else
                localResults = &tcd.results[localStartPoint - tcd.globalFirst];

            #ifdef DBG_TIME
                sw.stop();
                time_assign += sw.getMsec();
            #endif

            #ifdef DBG_DATA
                printf("[%d] ComputeThread %d: Got %lu elements starting from %lu to %lu\n",
                            rank, cti.id, localNumOfElements, localStartPoint, localLast);
                fflush(stdout);
            #else
                #ifdef DBG_START_STOP
                    printf("[%d] ComputeThread %d: Running for %lu elements...\n", rank, cti.id, localNumOfElements);
                    fflush(stdout);
                #endif
            #endif

            #ifdef DBG_TIME
                sw.start();
            #endif

            /*****************************************************************
             If batchSize was increased, allocate more memory for the results
            ******************************************************************/
            // TODO: Move this to initialization and allocate as much memory as possible
            if (allocatedElements < localNumOfElements && cti.id > -1 && allocatedElements < maxGpuBatchSize) {

                #ifdef DBG_MEMORY
                    printf("[%d] ComputeThread %d: Allocating more GPU memory (%lu", rank, cti.id, allocatedElements);
                    fflush(stdout);
                #endif

                allocatedElements = std::min(localNumOfElements, maxGpuBatchSize);

                #ifdef DBG_MEMORY
                    printf(" -> %lu elements, %lu MB)\n", allocatedElements, (allocatedElements*sizeof(RESULT_TYPE)) / (1024 * 1024));
                    fflush(stdout);
                #endif

                // Reallocate memory on device
                cudaFree(deviceResults);
                cce();
                cudaMalloc(&deviceResults, allocatedElements * sizeof(RESULT_TYPE));
                cce();

                #ifdef DBG_MEMORY
                    printf("[%d] ComputeThread %d: deviceResults = 0x%x\n", rank, cti.id, deviceResults);
                #endif
            }

            #ifdef DBG_TIME
                sw.stop();
                time_allocation += sw.getMsec();
                sw.start();
            #endif

            /*****************************************************************
            ******************** Calculate the results ***********************
            ******************************************************************/
            // If GPU...
            if (cti.id > -1) {

                // Initialize the list index counter
                cudaMemset(deviceListIndexPtr, 0, sizeof(int));

                // Divide the chunk to smaller chunks to scatter accross streams
                unsigned long elementsPerStream = localNumOfElements / parameters.gpuStreams;
                bool onlyOne = false;
                unsigned long skip = 0;
                if(elementsPerStream == 0){
                    elementsPerStream = localNumOfElements;
                    onlyOne = true;
                }

                // Queue the chunks to the streams
                for(int i=0; i<parameters.gpuStreams; i++){
                    // Adjust elementsPerStream for last stream (= total-queued)
                    if(i == parameters.gpuStreams - 1){
                        elementsPerStream = localNumOfElements - skip;
                    }else{
                        elementsPerStream = std::min(elementsPerStream, localNumOfElements - skip);
                    }

                    // Queue the kernel in stream[i] (each GPU thread gets COMPUTE_BATCH_SIZE elements to calculate)
                    int gpuThreads = (elementsPerStream + parameters.computeBatchSize - 1) / parameters.computeBatchSize;

                    // Minimum of (minimum of user-defined block size and number of threads to go to this stream) and number of points that can fit in shared memory
                    int blockSize = std::min(std::min(parameters.blockSize, gpuThreads), maxSharedPoints);
                    int numOfBlocks = (gpuThreads + blockSize - 1) / blockSize;

                    #ifdef DBG_QUEUE
                        printf("[%d] ComputeThread %d: Queueing %lu elements in stream %d (%d gpuThreads, %d blocks, %d block size), with skip=%lu\n", rank,
                                cti.id, elementsPerStream, i, gpuThreads, numOfBlocks, blockSize, skip);
                    #endif

                    // Note: Point at the start of deviceResults, because the offset (because of computeBatchSize) is calculated in the kernel
                    callGpuKernel(numOfBlocks, blockSize, deviceProp.sharedMemPerBlock, streams[i],
                        deviceResults, localStartPoint,
                        parameters.D, elementsPerStream, skip, deviceDataPtr,
                        parameters.dataSize, useSharedMemoryForData, useConstantMemoryForData,
                        parameters.resultSaveType == SAVE_TYPE_ALL ? nullptr : deviceListIndexPtr,
                        parameters.computeBatchSize
                    );

                    // Queue the memcpy in stream[i] only if we are saving as SAVE_TYPE_ALL (otherwise the results will be fetched at the end of the current computation)
                    if(parameters.resultSaveType == SAVE_TYPE_ALL){
                        cudaMemcpyAsync(&localResults[skip], &deviceResults[skip], elementsPerStream*sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost, streams[i]);
                    }

                    // Increase skip
                    skip += elementsPerStream;

                    if(onlyOne)
                        break;
                }

                #ifdef DBG_TIME
                    sw.stop();
                    time_queue += sw.getMsec();
                    sw.start();
                #endif

                // Wait for all streams to finish
                for(int i=0; i<parameters.gpuStreams; i++){
                    cudaStreamSynchronize(streams[i]);
                    cce();
                }

                // If we are saving as SAVE_TYPE_LIST, fetch the results
                if(parameters.resultSaveType == SAVE_TYPE_LIST){
                    // Get the current list index from the GPU
                    cudaMemcpy(&gpuListIndex, deviceListIndexPtr, sizeof(int), cudaMemcpyDeviceToHost);

                    // Increment the global list index counter
                    globalListIndexOld = __sync_fetch_and_add(&tcd.listIndex, gpuListIndex);

                    // Get the results from the GPU
                    cudaMemcpy(&((DATA_TYPE*)localResults)[globalListIndexOld], deviceResults, gpuListIndex * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
                }

                if(nvmlAvailable){
                    unsigned int tmpSamples;

                    // Get number of available samples
                    result = nvmlDeviceGetSamples(gpuHandle, NVML_GPU_UTILIZATION_SAMPLES, lastSeenTimeStamp, &sampleValType, &tmpSamples, NULL);
                    if (result != NVML_SUCCESS && result != NVML_ERROR_NOT_FOUND) {
                        printf("[%d] [E1] Failed to get utilization samples for device: %s\n", rank, nvmlErrorString(result));
                    }else if(result == NVML_SUCCESS){

                        // Make sure we have enough allocated memory for the new samples
                        if(tmpSamples > numOfAllocatedSamples){
                            delete[] samples;
                            samples = new nvmlSample_t[tmpSamples];
                        }

                        result = nvmlDeviceGetSamples(gpuHandle, NVML_GPU_UTILIZATION_SAMPLES, lastSeenTimeStamp, &sampleValType, &tmpSamples, samples);
                        if (result == NVML_SUCCESS) {
                            numOfSamples += tmpSamples;
                            for(unsigned int i=0; i<tmpSamples; i++){
                                totalUtilization += samples[i].sampleValue.uiVal;
                            }
                        }else if(result != NVML_ERROR_NOT_FOUND){
                            printf("[%d] [E2] Failed to get utilization samples for device: %s\n", rank, nvmlErrorString(result));
                        }
                    }

                }

            } else {

                #ifdef DBG_TIME
                    sw.stop();
                    time_queue += sw.getMsec();
                    sw.start();
                #endif

                callCpuKernel(localResults, limits.data(), parameters.D, localNumOfElements, parameters.dataPtr, parameters.resultSaveType == SAVE_TYPE_ALL ? nullptr : &tcd.listIndex,
                            idxSteps.data(), localStartPoint, parameters.cpuDynamicScheduling, parameters.cpuComputeBatchSize);

            }

            #ifdef DBG_TIME
                sw.stop();
                time_calc += sw.getMsec();
            #endif

            #ifdef DBG_RESULTS
                if(parameters.resultSaveType == SAVE_TYPE_ALL){
                    printf("[%d] ComputeThread %d: Results are: ", rank, cti.id);
                    for (unsigned long i = 0; i < localNumOfElements; i++) {
                        printf("%f ", ((DATA_TYPE *)localResults)[i]);
                    }
                    printf("\n");
                }
            #endif

            #ifdef DBG_START_STOP
                printf("[%d] ComputeThread %d: Finished calculation\n", rank, cti.id);
            #endif

            cti.elementsCalculated += localNumOfElements;

            if(!parameters.slaveDynamicScheduling)
                break;

            // End of assignment loop
        }

        // Stop the stopwatch
        cti.stopwatch.stop();

        #ifdef DBG_TIME
            printf("[%d] ComputeThread %d: Benchmark:\n", rank, cti.id);
            printf("Time for first sleep: %f ms\n", time_sleep);
            printf("Time for assignments: %f ms\n", time_assign);
            printf("Time for allocations: %f ms\n", time_allocation);
            printf("Time for queues: %f ms\n", time_queue);
            printf("Time for calcs: %f ms\n", time_calc);
        #endif

        // Let coordinatorThread know that the results are ready
        tcd.postResultsSemaphore();
    }

    #ifdef DBG_START_STOP
        printf("[%d] ComputeThread %d: Finalizing and exiting...\n", rank, cti.id);
    #endif

    if(cti.id >= 0 && nvmlAvailable)
        cti.averageUtilization = numOfSamples > 0 ? totalUtilization/numOfSamples : 0;

    if(cti.id == -1 && startUptime > 0 && startIdleTime > 0){
        if(getCpuStats(&endUptime, &endIdleTime) == 0){
            cti.averageUtilization = 100 - 100 * (endIdleTime - startIdleTime) / (endUptime - startUptime);
        }
    }

    /*******************************************************************
     *************************** Finalize ******************************
     *******************************************************************/
    // Finalize GPU
    if (cti.id > -1) {
        // Make sure streams are finished and destroy them
        for(int i=0;i<parameters.gpuStreams;i++){
            cudaStreamDestroy(streams[i]);
            cce();
        }

        // Deallocate device's memory
        cudaFree(deviceResults);			cce();
        cudaFree(deviceListIndexPtr);		cce();
        cudaFree(deviceDataPtr);			cce();

        delete[] samples;
        if(nvmlInitialized){
            result = nvmlShutdown();
            if (result != NVML_SUCCESS)
                printf("[%d] [E] Failed to shutdown NVML: %s\n", rank, nvmlErrorString(result));
        }
    }
}

