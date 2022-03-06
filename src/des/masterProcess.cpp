#include "framework.h"

#include <cstring>

void ParallelFramework::masterProcess() {
    int finished = 0;
    float totalScore;
    int t, eta;
    int numOfProcesses = getNumOfProcesses();
    int numOfSlaves = numOfProcesses - 1;
    Stopwatch masterStopwatch;

    SlaveProcessInfo slaveProcessInfo[numOfSlaves];

    void* tmpResultsMem;
    if(parameters.overrideMemoryRestrictions){
        tmpResultsMem = malloc(getMaxCPUBytes());
    }else{
        tmpResultsMem = malloc(parameters.resultSaveType == SAVE_TYPE_ALL ? parameters.batchSize * sizeof(RESULT_TYPE) : parameters.batchSize * parameters.D * sizeof(DATA_TYPE));
    }
    if(tmpResultsMem == nullptr){
        printf("[%d] Master: Error: Can't allocate memory for tmpResultsMem\n", rank);
        exit(-1);
    }
    RESULT_TYPE* tmpResults = (RESULT_TYPE*) tmpResultsMem;
    decltype(listResults) tmpResultsList;// = (DATA_TYPE*) tmpResultsMem;
    int tmpNumOfPoints;

    #ifdef DBG_TIME
        Stopwatch sw;
    #endif

    for(int i=0; i<numOfSlaves; i++){
        slaveProcessInfo[i].id = i + 1;
        slaveProcessInfo[i].currentBatchSize = parameters.batchSize;
        slaveProcessInfo[i].jobsCompleted = 0;
        slaveProcessInfo[i].elementsCalculated = 0;
        slaveProcessInfo[i].finished = false;
        slaveProcessInfo[i].lastScore = -1;
        slaveProcessInfo[i].ratio = (float)1/numOfSlaves;
    }

    masterStopwatch.start();
    while (totalReceived < totalElements || finished < numOfSlaves) {
        // Receive request from any worker thread
        #ifdef DBG_MPI_STEPS
            printf("[%d] Master: Waiting for signal...\n", rank);
        #endif
        fflush(stdout);

        int mpiSource;
        int request = receiveRequest(mpiSource);

        SlaveProcessInfo& pinfo = slaveProcessInfo[mpiSource-1];

        #ifdef DBG_MPI_STEPS
            printf("[%d] Master: Received %d from %d\n", rank, status.MPI_TAG, mpiSource);
        #endif

        switch(request){
            case TAG_READY:
                #ifdef DBG_TIME
                    sw.start();
                #endif
                // Receive the maximum batch size reported by the slave process
                pinfo.maxBatchSize = receiveMaxBatchSize(mpiSource);

                // For the first batches, use low batch size so the process can optimize its computeThread scores early
                if(pinfo.jobsCompleted < parameters.slowStartLimit){
                    pinfo.maxBatchSize = std::min(pinfo.maxBatchSize, (unsigned long) (parameters.slowStartBase * pow(2, pinfo.jobsCompleted)));
                    #ifdef DBG_RATIO
                        printf("[%d] Master: Setting temporary maxBatchSize=%lu for slave %d\n", rank, pinfo.maxBatchSize, mpiSource);
                    #endif
                }

                // Get next data batch to calculate
                if(totalSent == totalElements){
                    pinfo.work.startPoint = 0;
                    pinfo.work.numOfElements = 0;
                }else{
                    pinfo.work.startPoint = totalSent;
                    pinfo.work.numOfElements = std::min(std::min((unsigned long) (pinfo.ratio * parameters.batchSize), (unsigned long) pinfo.maxBatchSize), totalElements-totalSent);

                    // printf("pinfo.ratio = %f, paramters->batchSize = %lu, pinfo.maxBatchSize = %lu, totalElements = %lu, totalSent = %lu, product = %lu\n",
                    // 		pinfo.ratio, parameters.batchSize, pinfo.maxBatchSize, totalElements, totalSent, (unsigned long) (pinfo.ratio * parameters.batchSize));

                    totalSent += pinfo.work.numOfElements;
                }

                #ifdef DBG_MPI_STEPS
                    printf("[%d] Master: Sending %lu elements to %d with index %lu\n", rank, pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
                #endif
                #ifdef DBG_DATA
                    printf("[%d] Master: Sending %lu elements to %d with index %lu\n", rank, pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
                #endif

                // Send the batch to the slave process
                sendBatchSize(pinfo.work, mpiSource);

                // Start stopwatch for process
                pinfo.stopwatch.start();

                #ifdef DBG_TIME
                    sw.stop();
                    printf("[%d] Master: Benchmark: Time for TAG_READY: %f ms\n", rank, sw.getMsec());
                #endif

                break;

            case TAG_RESULTS:
                #ifdef DBG_TIME
                    sw.start();
                #endif

                // Receive the results
                if(parameters.resultSaveType == SAVE_TYPE_ALL){
                    receiveAllResults(tmpResults, pinfo.work.numOfElements, mpiSource);
                }else{
                    tmpNumOfPoints = receiveListResults((DATA_TYPE*) tmpResultsMem, pinfo.work.numOfElements, mpiSource);
                    for(int i=0; i<tmpNumOfPoints; i++){
                        std::vector<DATA_TYPE> point;
                        for(unsigned int j=0; j<parameters.D; j++){
                            point.push_back(((DATA_TYPE*)tmpResultsMem)[i*parameters.D + j]);
                        }
                        tmpResultsList.push_back(point);
                    }
                }

                this->totalReceived += pinfo.work.numOfElements;

                masterStopwatch.stop();
                t = masterStopwatch.getMsec()/1000;
                eta = t * ((float)totalElements/totalReceived) - t;

                if(parameters.printProgress){
                    printf("Progress: %lu/%lu, %.2f %%", this->totalReceived, this->totalElements, ((float)this->totalReceived / this->totalElements)*100);

                    if(t < 3600)	printf(", Elapsed time: %02d:%02d", t/60, t%60);
                    else			printf(", Elapsed time: %02d:%02d:%02d", t/3600, (t%3600)/60, t%60);

                    if(eta < 3600)	printf(", ETA: %02d:%02d\n", eta/60, eta%60);
                    else			printf(", ETA: %02d:%02d:%02d\n", eta/3600, (eta%3600)/60, eta%60);
                }

                #ifdef DBG_MPI_STEPS
                    if(parameters.resultSaveType == SAVE_TYPE_ALL)
                        printf("[%d] Master: Saving %lu results from slave %d to finalResults[%lu]...\n", rank, pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
                    else
                        printf("[%d] Master: Saving %lu points from slave %d to listResults[%lu]...\n", rank, tmpNumOfPoints, mpiSource, listResultsSaved);

                #endif
                #ifdef DBG_RESULTS
                    if(parameters.resultSaveType == SAVE_TYPE_ALL){
                        printf("[%d] Master: Saving tmpResults: ", rank);
                        for (int i = 0; i < pinfo.work.numOfElements; i++) {
                            printf("%f ", tmpResults[i]);
                        }
                        printf(" at %lu\n", pinfo.work.startPoint);
                    }else{
                        printf("[%d] Master: Saving tmpResultsList: ", rank);
                        for (int i = 0; i < tmpNumOfPoints; i++){
                            printf("[ ");
                            for(int j=0; j<parameters.D; j++){
                                printf("%f ", tmpResultsList[i*parameters.D + j]);
                            }
                            printf("]");
                        }
                        printf(" at %d\n", listResultsSaved);
                    }
                #endif

                // Copy the received results to finalResults or listResults
                if( ! (parameters.benchmark)){
                    if(parameters.resultSaveType == SAVE_TYPE_ALL){
                        memcpy(&finalResults[pinfo.work.startPoint], tmpResults, pinfo.work.numOfElements*sizeof(RESULT_TYPE));
                    }else if(tmpNumOfPoints > 0){
                        listResults.insert(listResults.end(), tmpResultsList.begin(), tmpResultsList.end());
                    }
                }

                // Update pinfo
                pinfo.jobsCompleted++;
                pinfo.elementsCalculated += pinfo.work.numOfElements;
                pinfo.stopwatch.stop();
                pinfo.lastScore = pinfo.work.numOfElements / pinfo.stopwatch.getMsec();
                pinfo.lastAssignedElements = pinfo.work.numOfElements;

                // Print benchmark results
                if (parameters.benchmark && parameters.printProgress) {
                    printf("[%d] Master: Slave %d benchmark: %lu elements, %f ms\n\n", rank, mpiSource, pinfo.work.numOfElements, pinfo.stopwatch.getMsec());
                }

                if(parameters.slaveBalancing && numOfSlaves > 1){
                    // Check other scores and calculate the sum
                    totalScore = 0;
                    for(int i=0; i<numOfSlaves; i++){
                        totalScore += slaveProcessInfo[i].lastScore;

                        if(slaveProcessInfo[i].lastScore <= 0){
                            // Either score has not been set, or got negative time
                            totalScore = -1;
                            break;
                        }
                    }

                    // If all the processes have a real score (has been set and not negative)
                    if(totalScore > 0){
                        for(int i=0; i<numOfSlaves; i++){
                            slaveProcessInfo[i].ratio = slaveProcessInfo[i].lastScore / totalScore;
                            #ifdef DBG_RATIO
                                printf("[%d] Master: Adjusting slave %d ratio = %f\n", rank, slaveProcessInfo[i].id, slaveProcessInfo[i].ratio);
                            #endif
                        }
                    }else{
                        #ifdef DBG_RATIO
                            printf("[%d] Master: Skipping ratio adjustment\n", rank);
                        #endif
                    }
                }

                // Reset pinfo
                pinfo.work.numOfElements = 0;

                #ifdef DBG_TIME
                    sw.stop();
                    printf("[%d] Master: Benchmark: Time for TAG_RESULTS: %f ms\n", rank, sw.getMsec());
                #endif

                break;

            case TAG_EXITING:
                #ifdef DBG_MPI_STEPS
                    printf("[%d] Master: Slave %d exiting\n", rank, mpiSource);
                #endif

                if(pinfo.work.numOfElements != 0){
                    printf("[%d] Master: Error: Slave %d exited with %lu assigned elements!!\n", rank, mpiSource, pinfo.work.numOfElements);
                }

                pinfo.work.startPoint = totalElements;
                pinfo.finished = true;

                finished++;

                break;
        }
    }

    free(tmpResultsMem);

    // Synchronize with the rest of the processes
    #ifdef DBG_START_STOP
        printf("[%d] Waiting in barrier...\n", rank);
    #endif
    syncWithSlaves();
    #ifdef DBG_START_STOP
        printf("[%d] Passed the barrier...\n", rank);
    #endif
}

