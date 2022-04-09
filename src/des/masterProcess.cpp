#include "framework.h"
#include "utilities.h"

#include <cstring>

void ParallelFramework::masterProcess() {
    int finished = 0;
    int numOfSlaves = getNumOfProcesses() - 1;

    SlaveProcessInfo slaveProcessInfo[numOfSlaves];

    void* tmpResultsMem;
    if(m_parameters.overrideMemoryRestrictions){
        tmpResultsMem = malloc(getMaxCPUBytes());
    }else{
        tmpResultsMem = malloc(m_parameters.resultSaveType == SAVE_TYPE_ALL ? m_parameters.batchSize * sizeof(RESULT_TYPE) : m_parameters.batchSize * m_parameters.D * sizeof(DATA_TYPE));
    }
    if(tmpResultsMem == nullptr){
        printf("[%d] Master: Error: Can't allocate memory for tmpResultsMem\n", m_rank);
        exit(-1);
    }

    #ifdef DBG_TIME
        Stopwatch sw;
    #endif

    for(int i=0; i<numOfSlaves; i++){
        slaveProcessInfo[i].id = i + 1;
        slaveProcessInfo[i].currentBatchSize = m_parameters.batchSize;
        slaveProcessInfo[i].jobsCompleted = 0;
        slaveProcessInfo[i].elementsCalculated = 0;
        slaveProcessInfo[i].finished = false;
        slaveProcessInfo[i].lastScore = -1;
        slaveProcessInfo[i].ratio = (float)1/numOfSlaves;
    }

    Stopwatch masterStopwatch;
    masterStopwatch.start();
    while (m_totalReceived < m_totalElements || finished < numOfSlaves) {
        // Receive request from any worker thread
        #ifdef DBG_MPI_STEPS
            printf("[%d] Master: Waiting for signal...\n", m_rank);
        #endif
        fflush(stdout);

        int mpiSource;
        int request = receiveRequest(mpiSource);

        SlaveProcessInfo& pinfo = slaveProcessInfo[mpiSource-1];

        #ifdef DBG_MPI_STEPS
            printf("[%d] Master: Received %d from %d\n", m_rank, request, mpiSource);
        #endif

        switch(request){
            case TAG_READY:
                #ifdef DBG_TIME
                    sw.start();
                #endif
                // Receive the maximum batch size reported by the slave process
                pinfo.maxBatchSize = receiveMaxBatchSize(mpiSource);

                // For the first batches, use low batch size so the process can optimize its computeThread scores early
                if((int) pinfo.jobsCompleted < m_parameters.slowStartLimit){
                    pinfo.maxBatchSize = std::min(pinfo.maxBatchSize, (unsigned long) (m_parameters.slowStartBase * pow(2, pinfo.jobsCompleted)));
                    #ifdef DBG_RATIO
                        printf("[%d] Master: Setting temporary maxBatchSize=%lu for slave %d\n", m_rank, pinfo.maxBatchSize, mpiSource);
                    #endif
                }

                // Get next data batch to calculate
                if(m_totalSent == m_totalElements){
                    pinfo.work.startPoint = 0;
                    pinfo.work.numOfElements = 0;
                }else{
                    pinfo.work.startPoint = m_totalSent;
                    pinfo.work.numOfElements = std::min(std::min((unsigned long) (pinfo.ratio * m_parameters.batchSize), (unsigned long) pinfo.maxBatchSize), m_totalElements-m_totalSent);

                    // printf("pinfo.ratio = %f, paramters->batchSize = %lu, pinfo.maxBatchSize = %lu, m_totalElements = %lu, m_totalSent = %lu, product = %lu\n",
                    // 		pinfo.ratio, m_parameters.batchSize, pinfo.maxBatchSize, m_totalElements, m_totalSent, (unsigned long) (pinfo.ratio * m_parameters.batchSize));

                    m_totalSent += pinfo.work.numOfElements;
                }

                #ifdef DBG_MPI_STEPS
                    printf("[%d] Master: Sending %lu elements to %d with index %lu\n", m_rank, pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
                #endif
                #ifdef DBG_DATA
                    printf("[%d] Master: Sending %lu elements to %d with index %lu\n", m_rank, pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
                #endif

                // Send the batch to the slave process
                sendBatchSize(pinfo.work, mpiSource);

                // Start stopwatch for process
                pinfo.stopwatch.start();

                #ifdef DBG_TIME
                    sw.stop();
                    printf("[%d] Master: Benchmark: Time for TAG_READY: %f ms\n", m_rank, sw.getMsec());
                #endif

                break;

            case TAG_RESULTS: {
                #ifdef DBG_TIME
                    sw.start();
                #endif

                // Receive the results
                if(m_parameters.resultSaveType == SAVE_TYPE_ALL){
                    receiveAllResults(&m_finalResults[pinfo.work.startPoint], pinfo.work.numOfElements, mpiSource);
                    #ifdef DBG_RESULTS
                        printf("[%d] Master: Received results from %d starting at %lu\n", m_rank, pinfo.id, pinfo.work.startPoint);
                    #endif
                    #ifdef DBG_RESULTS_RAW
                        printf("[%d] Master: Received results from %d starting at %lu: ", m_rank, pinfo.id, pinfo.work.startPoint);
                        for (unsigned long i = 0; i < pinfo.work.numOfElements; i++) {
                            printf("%.2f ", m_finalResults[pinfo.work.startPoint + i]);
                        }
                        printf("\n");
                    #endif
                }else{
                    auto count = receiveListResults((DATA_TYPE*) tmpResultsMem, pinfo.work.numOfElements, mpiSource);
                    #ifdef DBG_RESULTS_RAW
                        printf("[%d] Master: Received %d list results from %d: ", m_rank, count, pinfo.id);
                    #endif
                    for(int i=0; i<count; i++){
                        std::vector<DATA_TYPE> point;
                        for(unsigned int j=0; j<m_parameters.D; j++){
                            point.push_back(((DATA_TYPE*)tmpResultsMem)[i*m_parameters.D + j]);
                        }
                        m_listResults.push_back(point);
                        #ifdef DBG_RESULTS_RAW
                            printf("[ ");
                            for(const auto& v : point) printf("%.2f ", v);
                            printf("]");
                        #endif
                    }
                    #ifdef DBG_RESULTS_RAW
                        printf("\n");
                    #endif
                    #ifdef DBG_RESULTS
                        printf("[%d] Master: Received %u elements from %d, new total is %lu\n", rank, count, pinfo.id, listResults.size());
                    #endif
                }

                m_totalReceived += pinfo.work.numOfElements;

                masterStopwatch.stop();
                int t = masterStopwatch.getMsec()/1000;
                int eta = t * ((float)m_totalElements/m_totalReceived) - t;

                if(m_parameters.printProgress){
                    printf("Progress: %lu/%lu, %.2f %%", this->m_totalReceived, this->m_totalElements, ((float)this->m_totalReceived / this->m_totalElements)*100);

                    if(t < 3600)	printf(", Elapsed time: %02d:%02d", t/60, t%60);
                    else			printf(", Elapsed time: %02d:%02d:%02d", t/3600, (t%3600)/60, t%60);

                    if(eta < 3600)	printf(", ETA: %02d:%02d\n", eta/60, eta%60);
                    else			printf(", ETA: %02d:%02d:%02d\n", eta/3600, (eta%3600)/60, eta%60);
                }

                #ifdef DBG_MPI_STEPS
                    if(m_parameters.resultSaveType == SAVE_TYPE_ALL)
                        printf("[%d] Master: Received results from slave %d\n", m_rank, mpiSource);
                    else
                        printf("[%d] Master: Received list results from slave %d\n", m_rank, mpiSource);
                #endif

                // Update pinfo
                pinfo.jobsCompleted++;
                pinfo.elementsCalculated += pinfo.work.numOfElements;
                pinfo.stopwatch.stop();
                pinfo.lastScore = pinfo.work.numOfElements / pinfo.stopwatch.getMsec();
                pinfo.lastAssignedElements = pinfo.work.numOfElements;

                // Print benchmark results
                if (m_parameters.benchmark && m_parameters.printProgress) {
                    printf("[%d] Master: Slave %d benchmark: %lu elements, %f ms\n\n", m_rank, mpiSource, pinfo.work.numOfElements, pinfo.stopwatch.getMsec());
                }

                if(m_parameters.slaveBalancing && numOfSlaves > 1){
                    // Check other scores and calculate the sum
                    float totalScore = 0;
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
                                printf("[%d] Master: Adjusting slave %d ratio = %f\n", m_rank, slaveProcessInfo[i].id, slaveProcessInfo[i].ratio);
                            #endif
                        }
                    }else{
                        #ifdef DBG_RATIO
                            printf("[%d] Master: Skipping ratio adjustment\n", m_rank);
                        #endif
                    }
                }

                // Reset pinfo
                pinfo.work.numOfElements = 0;

                #ifdef DBG_TIME
                    sw.stop();
                    printf("[%d] Master: Benchmark: Time for TAG_RESULTS: %f ms\n", m_rank, sw.getMsec());
                #endif

                break;
            }

            case TAG_EXITING:
                #ifdef DBG_MPI_STEPS
                    printf("[%d] Master: Slave %d exiting\n", m_rank, mpiSource);
                #endif

                if(pinfo.work.numOfElements != 0){
                    printf("[%d] Master: Error: Slave %d exited with %lu assigned elements!!\n", m_rank, mpiSource, pinfo.work.numOfElements);
                }

                pinfo.work.startPoint = m_totalElements;
                pinfo.finished = true;

                finished++;

                break;
        }
    }

    free(tmpResultsMem);

    // Synchronize with the rest of the processes
    #ifdef DBG_START_STOP
        printf("[%d] Waiting in barrier...\n", m_rank);
    #endif
    syncWithSlaves();
    #ifdef DBG_START_STOP
        printf("[%d] Passed the barrier...\n", m_rank);
    #endif
}

