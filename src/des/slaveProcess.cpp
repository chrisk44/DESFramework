#include "desf.h"
#include "coordinatorThread.h"

#include <list>

namespace desf {

void DesFramework::slaveProcess() {
    Stopwatch masterStopwatch;
    masterStopwatch.start();

    // Calculate number of worker threads (#GPUs + 1CPU)
    bool useCpu = m_config.processingType == PROCESSING_TYPE_CPU || m_config.processingType == PROCESSING_TYPE_BOTH;
    bool useGpu = m_config.processingType == PROCESSING_TYPE_GPU || m_config.processingType == PROCESSING_TYPE_BOTH;
    int numOfCpus = useCpu ? 1 : 0;
    int numOfGpus = 0;

    if(useGpu){
        cudaGetDeviceCount(&numOfGpus);
        if(numOfGpus == 0){
            if(useCpu){
                log("Warning: cudaGetDeviceCount returned 0. Will use only CPU(s).");
            } else {
                log("Error: cudaGetDeviceCount returned 0. Exiting.");
                return;
            }
        }
    }

    std::vector<ComputeThread> computeThreads;
    for(int i=0; i<numOfCpus; i++) computeThreads.emplace_back(-i-1, "CPU" + std::to_string(i), WorkerThreadType::CPU, getConfig(), getIndexSteps());
    for(int i=0; i<numOfGpus; i++) computeThreads.emplace_back(i,  "GPU" + std::to_string(i), WorkerThreadType::GPU, getConfig(), getIndexSteps());

    #ifdef DBG_TIME
        masterStopwatch.stop();
        log("Time to create compute threads: %f ms", masterStopwatch.getMsec());
    #endif

    #ifdef DBG_START_STOP
        log("Created %lu compute threads", computeThreads.size());
    #endif

    CoordinatorThread coordinatorThread(getConfig(), computeThreads);
    coordinatorThread.run();

    // Notify master about exiting
    sendExitSignal();

    masterStopwatch.stop();
//    float masterTime = masterStopwatch.getMsec();
    for(auto& cti : computeThreads){
        // if(cti.averageUtilization >= 0){
//            float resourceTime = cti.masterStopwatch.getMsec();
//            float finishIdleTime = masterTime - resourceTime;
//            cti.idleTime += finishIdleTime;
            log("Resource %d utilization: %.02f%%, total idle time: %.02f%% (%.02fms) (%s)",
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
