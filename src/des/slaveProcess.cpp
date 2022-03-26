#include "framework.h"

void ParallelFramework::slaveProcessImpl(CallComputeThreadCallback callComputeThread) {
    /*******************************************************************
    ********** Calculate number of worker threads (#GPUs + 1CPU) *******
    ********************************************************************/
    Stopwatch masterStopwatch;
    masterStopwatch.start();

    bool useCpu = parameters.processingType == PROCESSING_TYPE_CPU || parameters.processingType == PROCESSING_TYPE_BOTH;
    bool useGpu = parameters.processingType == PROCESSING_TYPE_GPU || parameters.processingType == PROCESSING_TYPE_BOTH;
    int numOfCpus = useCpu ? 1 : 0;
    int numOfGpus = 0;

    if(useGpu){
        cudaGetDeviceCount(&numOfGpus);
        if(numOfGpus == 0){
            if(useCpu){
                printf("[%d] SlaveProcess: Warning: cudaGetDeviceCount returned 0. Will use only CPU(s).\n", rank);
            } else {
                printf("[%d] SlaveProcess: Error: cudaGetDeviceCount returned 0. Exiting.", rank);
                valid = false;
                return;
            }
        }
    }

    /*******************************************************************
    ********************** Initialization ******************************
    ********************************************************************/
    ThreadCommonData threadCommonData;

    std::vector<ComputeThreadInfo> computeThreadInfo;
    for(int i=0; i<numOfCpus; i++) computeThreadInfo.emplace_back(-i-1, "CPU" + std::to_string(i), 0, numOfCpus + numOfGpus);
    for(int i=0; i<numOfGpus; i++) computeThreadInfo.emplace_back(i,  "GPU" + std::to_string(i), 0, numOfCpus + numOfGpus);

    #ifdef DBG_START_STOP
        printf("[%d] SlaveProcess: Spawning %d worker threads...\n", rank, numOfThreads);
    #endif

    /*******************************************************************
    *************** Launch coordinator and worker threads **************
    ********************************************************************/
    #pragma omp parallel num_threads(computeThreadInfo.size() + 1) shared(computeThreadInfo) 	// +1 thread to handle the communication with masterProcess
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            coordinatorThread(computeThreadInfo, threadCommonData);
        }else{
            auto& cti = computeThreadInfo[tid-1];
            cti.masterStopwatch.start();
            callComputeThread(cti, threadCommonData);
            cti.masterStopwatch.stop();
        }
    }

    // Synchronize with the rest of the processes
    #ifdef DBG_START_STOP
        printf("[%d] Syncing with other slave processes...\n", rank);
    #endif
    syncWithSlaves();
    #ifdef DBG_START_STOP
        printf("[%d] Synced with other slave processes...\n", rank);
    #endif

    masterStopwatch.stop();
    float masterTime = masterStopwatch.getMsec();
    for(auto& cti : computeThreadInfo){
        // if(cti.averageUtilization >= 0){
            float resourceTime = cti.masterStopwatch.getMsec();
            float finishIdleTime = masterTime - resourceTime;
            cti.idleTime += finishIdleTime;
            printf("[%d] Resource %d utilization: %.02f%%, total idle time: %.02f%% (%.02fms) (%s)\n", rank,
                    cti.id,
                    cti.averageUtilization,
                    (cti.idleTime / masterTime) * 100,
                    cti.idleTime,
                    cti.name.size() == 0 ? "unnamed" : cti.name.c_str()
            );
        // }
    }

}

