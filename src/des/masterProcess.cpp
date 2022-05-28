#include "desf.h"
#include "utilities.h"

#include <cstring>

namespace desf {

void DesFramework::masterProcess() {
    int finished = 0;
    int numOfSlaves = getNumOfProcesses() - 1;

    SlaveProcessInfo slaveProcessInfo[numOfSlaves];

    std::vector<DATA_TYPE> tmpList;

    #ifdef DBG_TIME
        Stopwatch sw;
    #endif

    for(int i=0; i<numOfSlaves; i++){
        slaveProcessInfo[i].id = i + 1;
        slaveProcessInfo[i].currentBatchSize = m_config.batchSize;
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
            log("Waiting for signal...");
        #endif
        fflush(stdout);

        int mpiSource;
        int request = receiveRequest(mpiSource);

        SlaveProcessInfo& pinfo = slaveProcessInfo[mpiSource-1];

        #ifdef DBG_MPI_STEPS
            log("Received tag '%s' from %d", TAG_NAMES.at(request).c_str(), mpiSource);
        #endif

        switch(request){
            case TAG_READY:
                #ifdef DBG_TIME
                    sw.start();
                #endif
                // Receive the maximum batch size reported by the slave process
                pinfo.maxBatchSize = receiveMaxBatchSize(mpiSource);

                // For the first batches, use low batch size so the process can optimize its computeThread scores early
                if((int) pinfo.jobsCompleted < m_config.slowStartLimit){
                    pinfo.maxBatchSize = std::min(pinfo.maxBatchSize, (unsigned long) (m_config.slowStartBase * pow(2, pinfo.jobsCompleted)));
                    #ifdef DBG_RATIO
                        log("Setting temporary maxBatchSize=%lu for slave %d", pinfo.maxBatchSize, mpiSource);
                    #endif
                }

                // Get next data batch to calculate
                if(m_totalSent == m_totalElements){
                    pinfo.work.startPoint = 0;
                    pinfo.work.numOfElements = 0;
                }else{
                    pinfo.work.startPoint = m_totalSent;
                    pinfo.work.numOfElements = std::min(std::min((unsigned long) (pinfo.ratio * m_config.batchSize), (unsigned long) pinfo.maxBatchSize), m_totalElements-m_totalSent);

                    // printf("pinfo.ratio = %f, paramters->batchSize = %lu, pinfo.maxBatchSize = %lu, m_totalElements = %lu, m_totalSent = %lu, product = %lu\n",
                    // 		pinfo.ratio, m_config.batchSize, pinfo.maxBatchSize, m_totalElements, m_totalSent, (unsigned long) (pinfo.ratio * m_config.batchSize));

                    m_totalSent += pinfo.work.numOfElements;
                }

                #ifdef DBG_MPI_STEPS
                    log("Sending %lu elements to %d with index %lu", pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
                #endif
                #ifdef DBG_DATA
                    log("Sending %lu elements to %d with index %lu", pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
                #endif

                // Send the batch to the slave process
                sendBatchSize(pinfo.work, mpiSource);

                // Start stopwatch for process
                pinfo.stopwatch.start();

                #ifdef DBG_TIME
                    sw.stop();
                    log("Benchmark: Time for TAG_READY: %f ms", sw.getMsec());
                #endif

                break;

            case TAG_RESULTS: {
                #ifdef DBG_TIME
                    sw.start();
                #endif

                // Receive the results
                if(m_config.resultSaveType == SAVE_TYPE_ALL){
                    receiveAllResults(&m_finalResults[pinfo.work.startPoint], pinfo.work.numOfElements, mpiSource);
                    #ifdef DBG_RESULTS
                        log("Received results from %d starting at %lu", pinfo.id, pinfo.work.startPoint);
                    #endif
                    #ifdef DBG_RESULTS_RAW
                        std::string str;
                        for (unsigned long i = 0; i < pinfo.work.numOfElements; i++) {
                            char tmp[64];
                            sprintf(tmp, "%.2f ", m_finalResults[pinfo.work.startPoint + i]);
                            str += tmp;
                        }
                        log("Received results from %d starting at %lu: %s", m_rank, pinfo.id, pinfo.work.startPoint, str.c_str());
                    #endif
                }else{
                    auto count = receiveListResults(tmpList, pinfo.work.numOfElements, m_config.model.D, mpiSource);
                    #ifdef DBG_RESULTS_RAW
                        log("Received %d list results from %d: ", m_rank, count, pinfo.id);
                    #endif
                    for(int i=0; i<count; i++){
                        std::vector<DATA_TYPE> point;
                        for(unsigned int j=0; j<m_config.model.D; j++){
                            point.push_back(tmpList[i*m_config.model.D + j]);
                        }
                        m_listResults.push_back(point);
                        #ifdef DBG_RESULTS_RAW
                            std::string str = "[ ";
                            for(const auto& v : point){
                                char tmp[64];
                                sprintf(tmp, "%.2f ", v);
                                str += tmp;
                            }
                            str += "]";
                            log("Raw results: %s", str.c_str());
                        #endif
                    }
                    #ifdef DBG_RESULTS
                        log("Received %u elements from %d, new total is %lu", count, pinfo.id, m_listResults.size());
                    #endif
                }

                m_totalReceived += pinfo.work.numOfElements;

                masterStopwatch.stop();
                int t = masterStopwatch.getMsec()/1000;
                int eta = t * ((float)m_totalElements/m_totalReceived) - t;

                if(m_config.printProgress){
                    char elapsedTimeStr[1024];
                    char etaStr[1024];

                    if(t < 3600)	sprintf(elapsedTimeStr, "%02d:%02d", t/60, t%60);
                    else			sprintf(elapsedTimeStr, "%02d:%02d:%02d", t/3600, (t%3600)/60, t%60);

                    if(eta < 3600)	sprintf(etaStr, "%02d:%02d\n", eta/60, eta%60);
                    else			sprintf(etaStr, "%02d:%02d:%02d\n", eta/3600, (eta%3600)/60, eta%60);

                    log("Progress: %lu/%lu, %.2f %%, Elapsed time: %s, ETA: %s", m_totalReceived, m_totalElements, ((float) m_totalReceived / m_totalElements)*100, elapsedTimeStr, etaStr);

                }

                #ifdef DBG_MPI_STEPS
                    if(m_config.resultSaveType == SAVE_TYPE_ALL)
                        log("Received results from slave %d", mpiSource);
                    else
                        log("Received list results from slave %d", mpiSource);
                #endif

                // Update pinfo
                pinfo.jobsCompleted++;
                pinfo.elementsCalculated += pinfo.work.numOfElements;
                pinfo.stopwatch.stop();
                pinfo.lastScore = pinfo.work.numOfElements / pinfo.stopwatch.getMsec();
                pinfo.lastAssignedElements = pinfo.work.numOfElements;

                // Print benchmark results
                if (m_config.benchmark && m_config.printProgress) {
                    log("Slave %d benchmark: %lu elements, %f ms", mpiSource, pinfo.work.numOfElements, pinfo.stopwatch.getMsec());
                }

                if(m_config.slaveBalancing && numOfSlaves > 1){
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
                                log("Adjusting slave %d ratio = %f", slaveProcessInfo[i].id, slaveProcessInfo[i].ratio);
                            #endif
                        }
                    }else{
                        #ifdef DBG_RATIO
                            log("Skipping ratio adjustment");
                        #endif
                    }
                }

                // Reset pinfo
                pinfo.work.numOfElements = 0;

                #ifdef DBG_TIME
                    sw.stop();
                    log("Benchmark: Time for TAG_RESULTS: %f ms", sw.getMsec());
                #endif

                break;
            }

            case TAG_EXITING:
                #ifdef DBG_MPI_STEPS
                    log("Slave %d exiting", mpiSource);
                #endif

                if(pinfo.work.numOfElements != 0){
                    log("Error: Slave %d exited with %lu assigned elements!!", mpiSource, pinfo.work.numOfElements);
                }

                pinfo.work.startPoint = m_totalElements;
                pinfo.finished = true;

                finished++;

                break;
        }
    }

}

}
