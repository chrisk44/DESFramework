#include "desf.h"

#include <list>

namespace desf {

void DesFramework::slaveProcess() {
    /*******************************************************************
    ********** Calculate number of worker threads (#GPUs + 1CPU) *******
    ********************************************************************/
    Stopwatch masterStopwatch;
    masterStopwatch.start();

    bool useCpu = m_config.processingType == PROCESSING_TYPE_CPU || m_config.processingType == PROCESSING_TYPE_BOTH;
    bool useGpu = m_config.processingType == PROCESSING_TYPE_GPU || m_config.processingType == PROCESSING_TYPE_BOTH;
    int numOfCpus = useCpu ? 1 : 0;
    int numOfGpus = 0;

    if(useGpu){
        cudaGetDeviceCount(&numOfGpus);
        if(numOfGpus == 0){
            if(useCpu){
                printf("[%d] SlaveProcess: Warning: cudaGetDeviceCount returned 0. Will use only CPU(s).\n", m_rank);
            } else {
                printf("[%d] SlaveProcess: Error: cudaGetDeviceCount returned 0. Exiting.", m_rank);
                return;
            }
        }
    }

    /*******************************************************************
    ********************** Initialization ******************************
    ********************************************************************/
    ThreadCommonData tcd;

    std::vector<ComputeThread> computeThreads;
    for(int i=0; i<numOfCpus; i++) computeThreads.emplace_back(-i-1, "CPU" + std::to_string(i), WorkerThreadType::CPU, tcd, getConfig(), getIndexSteps());
    for(int i=0; i<numOfGpus; i++) computeThreads.emplace_back(i,  "GPU" + std::to_string(i), WorkerThreadType::GPU, tcd, getConfig(), getIndexSteps());

    #ifdef DBG_START_STOP
        printf("[%d] SlaveProcess: Created %lu compute threads...\n", m_rank, computeThreads.size());
    #endif

    coordinatorThread(computeThreads, tcd);

    // Notify master about exiting
    sendExitSignal();

    // Synchronize with the rest of the processes
    #ifdef DBG_START_STOP
        printf("[%d] Syncing with other slave processes...\n", m_rank);
    #endif
    syncWithSlaves();
    #ifdef DBG_START_STOP
        printf("[%d] Synced with other slave processes...\n", m_rank);
    #endif

    masterStopwatch.stop();
//    float masterTime = masterStopwatch.getMsec();
    for(auto& cti : computeThreads){
        // if(cti.averageUtilization >= 0){
//            float resourceTime = cti.masterStopwatch.getMsec();
//            float finishIdleTime = masterTime - resourceTime;
//            cti.idleTime += finishIdleTime;
            printf("[%d] Resource %d utilization: %.02f%%, total idle time: %.02f%% (%.02fms) (%s)\n", m_rank,
                    cti.getId(),                                // cti.id,
                    cti.getUtilization(),                       // cti.averageUtilization,
                    cti.getIdleTime() / cti.getTotalTime(),     // (cti.idleTime / masterTime) * 100,
                    cti.getIdleTime(),                          // cti.idleTime,
                    cti.getName().c_str()                       // cti.name.size() == 0 ? "unnamed" : cti.name.c_str()
            );
        // }
    }
}

}
