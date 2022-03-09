#include "framework.h"

void ParallelFramework::slaveProcessImpl(CallComputeThreadCallback callComputeThread) {
    /*******************************************************************
    ********** Calculate number of worker threads (#GPUs + 1CPU) *******
    ********************************************************************/
    int numOfThreads = 0;
    ComputeThreadInfo* computeThreadInfo;
    ThreadCommonData threadCommonData;
    Stopwatch masterStopwatch;
    masterStopwatch.start();

    if(parameters.processingType != PROCESSING_TYPE_CPU)
        cudaGetDeviceCount(&numOfThreads);

    if(parameters.processingType != PROCESSING_TYPE_GPU)
        numOfThreads++;

    if(numOfThreads > 0){

        /*******************************************************************
        ********************** Initialization ******************************
        ********************************************************************/

        sem_init(&threadCommonData.semResults, 0, 0);
        sem_init(&threadCommonData.semSync, 0, 1);
        threadCommonData.results = nullptr;

        computeThreadInfo = new ComputeThreadInfo[numOfThreads];
        for(int i=0; i<numOfThreads; i++){
            sem_init(&computeThreadInfo[i].semStart, 0, 0);
            computeThreadInfo[i].ratio = (float)1/numOfThreads;
            computeThreadInfo[i].totalRatio = 0;
            computeThreadInfo[i].name[0] = '\0';
        }

        #ifdef DBG_START_STOP
            printf("[%d] SlaveProcess: Spawning %d worker threads...\n", rank, numOfThreads);
        #endif

        /*******************************************************************
        *************** Launch coordinator and worker threads **************
        ********************************************************************/
        #pragma omp parallel num_threads(numOfThreads + 1) shared(computeThreadInfo) 	// +1 thread to handle the communication with masterProcess
        {
            int tid = omp_get_thread_num();
            if(tid == 0){
                coordinatorThread(computeThreadInfo, &threadCommonData, omp_get_num_threads()-1);
            }else{
                // Calculate id: -1 -> CPU, 0+ -> GPU[id]
                computeThreadInfo[tid-1].id = tid - (parameters.processingType == PROCESSING_TYPE_GPU ? 1 : 2);

                computeThreadInfo[tid-1].masterStopwatch.start();
                callComputeThread(computeThreadInfo[tid - 1], &threadCommonData);
                computeThreadInfo[tid-1].masterStopwatch.stop();
            }
        }
    } else {
        printf("[%d] SlaveProcess: Error: cudaGetDeviceCount returned 0\n", rank);
        valid = false;
    }

    // Synchronize with the rest of the processes
    #ifdef DBG_START_STOP
        printf("[%d] Waiting in barrier...\n", rank);
    #endif
    // MPI_Barrier(MPI_COMM_WORLD);
    int a = 0;
    MPI_Bcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    #ifdef DBG_START_STOP
        printf("[%d] Passed the barrier...\n", rank);
    #endif

    masterStopwatch.stop();
    float masterTime = masterStopwatch.getMsec();
    for(int i=0; i<numOfThreads; i++){
        // if(computeThreadInfo[i].averageUtilization >= 0){
            float resourceTime = computeThreadInfo[i].masterStopwatch.getMsec();
            float finishIdleTime = masterTime - resourceTime;
            computeThreadInfo[i].idleTime += finishIdleTime;
            printf("[%d] Resource %d utilization: %.02f%%, total idle time: %.02f%% (%.02fms) (%s)\n", rank,
                    computeThreadInfo[i].id,
                    computeThreadInfo[i].averageUtilization,
                    (computeThreadInfo[i].idleTime / masterTime) * 100,
                    computeThreadInfo[i].idleTime,
                    computeThreadInfo[i].name.size() == 0 ? "unnamed" : computeThreadInfo[i].name.c_str()
            );
        // }
    }

    /*******************************************************************
    ***************************** Finalize *****************************
    ********************************************************************/
    sem_destroy(&threadCommonData.semResults);
    sem_destroy(&threadCommonData.semSync);
    for(int i=0; i<numOfThreads; i++){
        sem_destroy(&computeThreadInfo[i].semStart);
    }
    delete[] computeThreadInfo;
}

