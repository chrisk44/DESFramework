#include "desf.h"
#include "scheduler/scheduler.h"
#include "utilities.h"

#include <cstring>

namespace desf {

void DesFramework::masterProcess() {
    int finished = 0;
    int numOfSlaves = getNumOfProcesses() - 1;

    if(numOfSlaves <= 0) {
        throw std::runtime_error("Only 1 process in MPI communicator, need at least 2");
    }

    SlaveProcessInfo slaveProcessInfo[numOfSlaves];

    std::vector<size_t> tmpList;

    #ifdef DBG_TIME
        Stopwatch sw;
    #endif

    std::vector<int> slaveProcessIds;
    for(int i=0; i<numOfSlaves; i++){
        slaveProcessInfo[i].id = i + 1;
        slaveProcessInfo[i].jobsCompleted = 0;
        slaveProcessInfo[i].elementsCalculated = 0;
        slaveProcessInfo[i].finished = false;

        slaveProcessIds.push_back(slaveProcessInfo[i].id);
    }

    auto maxBatchSizes = receiveMaxBatchSizes();

#ifdef DBG_DATA
    log("Received %lu max batch sizes:\n", maxBatchSizes.size());
    for(const auto& pair : maxBatchSizes) {
        log("Max batch size for %d is %lu", pair.first, pair.second);
    }
#endif

    m_config.interNodeScheduler->init(m_config, slaveProcessIds, maxBatchSizes);
    m_config.interNodeScheduler->setWork(AssignedWork(0, m_totalElements));

    Stopwatch masterStopwatch;
    masterStopwatch.start();
    while (finished < numOfSlaves) {
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

                pinfo.work = m_config.interNodeScheduler->getNextBatch(pinfo.id);

                #ifdef DBG_MPI_STEPS
                    log("Sending %lu elements to %d with index %lu", pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
                #endif
                #ifdef DBG_DATA
                    log("Sending %lu elements to %d with index %lu", pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
                #endif

                // Send the batch to the slave process
                sendBatch(pinfo.work, mpiSource);

                m_config.interNodeScheduler->onNodeStarted(pinfo.id);

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
                    receiveAllResults(&((RESULT_TYPE*) m_finalResults)[pinfo.work.startPoint], pinfo.work.numOfElements, mpiSource);
                    #ifdef DBG_RESULTS
                        log("Received results from %d starting at %lu", pinfo.id, pinfo.work.startPoint);
                    #endif
                    #ifdef DBG_RESULTS_RAW
                        std::string str;
                        for (unsigned long i = 0; i < pinfo.work.numOfElements; i++) {
                            char tmp[64];
                            sprintf(tmp, "%.2f ", ((RESULT_TYPE*) m_finalResults)[pinfo.work.startPoint + i]);
                            str += tmp;
                        }
                        log("Received results from %d starting at %lu: %s", m_rank, pinfo.id, pinfo.work.startPoint, str.c_str());
                    #endif
                }else{
                    auto count = receiveListResults(tmpList, pinfo.work.numOfElements, mpiSource);
                    #ifdef DBG_RESULTS_RAW
                        log("Received %d list results from %d: ", m_rank, count, pinfo.id);
                    #endif
                    for(int i=0; i<count; i++){
                        size_t index = tmpList[i];
                        std::vector<DATA_TYPE> point(m_config.model.D);
                        getPointFromIndex(index, point.data());
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

                m_config.interNodeScheduler->onNodeFinished(pinfo.id, pinfo.work.numOfElements, pinfo.stopwatch.getMsec());

                // Print benchmark results
                if (m_config.benchmark && m_config.printProgress) {
                    log("Slave %d benchmark: %lu elements, %f ms", mpiSource, pinfo.work.numOfElements, pinfo.stopwatch.getMsec());
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

                pinfo.finished = true;

                finished++;

                break;
        }
    }

    if(m_totalReceived != m_totalElements)
        throw std::runtime_error("Slave processes finished but not all elements have been calculated (received " +
                                 std::to_string(m_totalReceived) + "/" + std::to_string(m_totalElements) + ")");

    m_config.interNodeScheduler->finalize();
}

}
