#include "desf.h"

#include "coordinatorThread.h"
#include "utilities.h"

#include <stdarg.h>

namespace desf {

CoordinatorThread::CoordinatorThread(const DesConfig& config)
    : m_config(config),
      m_maxBatchSize(calculateMaxBatchSize(config)),
      m_maxCpuBatchSize(calculateMaxCpuBatchSize(config))
{
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
}

CoordinatorThread::~CoordinatorThread()
{}

void CoordinatorThread::log(const char *text, ...) {
    static thread_local char buf[65536];

    va_list args;
    va_start(args, text);
    vsnprintf(buf, sizeof(buf), text, args);
    va_end(args);

    printf("[%d] Coordinator: %s\n", m_rank, buf);
}

AssignedWork CoordinatorThread::getWork(ComputeThreadID threadID)
{
    unsigned long batchSize = m_batchSizes[threadID];
    AssignedWork work;
    {
        std::lock_guard<std::mutex> lock(m_syncMutex);  // TODO: Mutex is not needed since we are using an atomic, but we need to solve the overflow problem to discard the mutex

        // Fetch and increment the global batch start point by our batch size
        work.startPoint = m_globalBatchStart.fetch_add(batchSize);

        // Check for globalBatchStart overflow and limit it to globalLast+1 to avoid later overflows
        // If the new globalBatchStart is smaller than our local start point, the increment caused an overflow
        // If the localStart point in larger than the global last, then the elements have already been exhausted
        if(m_globalBatchStart < work.startPoint || work.startPoint > m_globalLast){
            // log("Fixing globalBatchStart from %lu to %lu", globalBatchStart, globalLast + 1);
            m_globalBatchStart = m_globalLast + 1;
        }
    }

    if(work.startPoint > m_globalLast){
        work.startPoint = 0;
        work.numOfElements = 0;
    } else {
        size_t last = std::min(work.startPoint + batchSize - 1 , m_globalLast);
        work.numOfElements = last - work.startPoint + 1;
    }

    return work;
}

void CoordinatorThread::run(std::vector<ComputeThread>& threads){
    WorkDispatcher workDispatcher = [&](ComputeThreadID id){ return getWork(id); };

    int numOfRatioAdjustments = 0;

    std::map<ComputeThreadID, float> ratios;
    std::map<ComputeThreadID, float> ratioSums;
    for(const auto& ct : threads){
        ratios[ct.getId()] = 1.f / (float) threads.size();
        ratioSums[ct.getId()] = 0.f;
    }

	#ifdef DBG_TIME
		Stopwatch sw;
		float time_data, time_assign, time_start, time_wait, time_scores, time_results;
    #endif

	#ifdef DBG_START_STOP
        log("Max batch size: %lu", m_maxBatchSize);
	#endif

	while(true){
		// Send READY signal to master
		#ifdef DBG_MPI_STEPS
            log("Sending READY...");
		#endif
		#ifdef DBG_TIME
			sw.start();
		#endif

        // Get a batch of data from master
        DesFramework::sendReadyRequest(m_maxBatchSize);
        AssignedWork work = DesFramework::receiveWorkFromMaster();

		#ifdef DBG_DATA
            log("Received %lu elements starting from %lu", work.numOfElements, work.startPoint);
		#endif
		#ifdef DBG_MPI_STEPS
            log("Received %lu elements", work.numOfElements);
		#endif

		// If no elements, break
		if(work.numOfElements == 0)
			break;

		// Make sure we have enough allocated memory for localResults
        if(work.numOfElements > m_results.size() && m_results.size() < m_maxCpuBatchSize){
            unsigned long newSize = std::min(work.numOfElements, m_maxCpuBatchSize);

            #ifdef DBG_MEMORY
                log("Allocating more memory for results: %lu -> %lu", m_results.size(), newSize);
            #endif

            m_results.resize(newSize);
            if(m_results.size() < newSize){
                fatal("Can't allocate memory for results");
			}
		}

		#ifdef DBG_TIME
			sw.stop();
			time_data = sw.getMsec();
			sw.start();
		#endif

		/*
		 * Split the data into pieces for each thread
		 * If we are using dynamic scheduling, each batch size will be
		 * ratio * min(slaveBatchSize, numOfElements).
		 * Otherwise it will be ratio * work.numOfElements and we will make sure
		 * later that exactly work.numOfElements elements have been assigned.
         */
        m_batchSizes.clear();
        std::map<ComputeThreadID, RESULT_TYPE*> resultPtrs;
        unsigned long total = 0;
        for(auto& ct : threads){
            if(m_config.slaveDynamicScheduling)
                m_batchSizes[ct.getId()] = std::max((unsigned long) 1, (unsigned long) (ratios[ct.getId()] * std::min(
                    m_config.slaveBatchSize, work.numOfElements)));
            else{
                m_batchSizes[ct.getId()] = std::max((unsigned long) 1, (unsigned long) (ratios[ct.getId()] * work.numOfElements));
                total += m_batchSizes[ct.getId()];
            }

            if(m_config.resultSaveType == SAVE_TYPE_LIST) {
                resultPtrs[ct.getId()] = m_results.data();
            } else {
                resultPtrs[ct.getId()] = &m_results[work.startPoint - m_globalFirst];
            }
        }

        /*
         * If we are NOT using dynamic scheduling, make sure that exactly
         * work.numOfElements elements have been assigned
         */
        if(!m_config.slaveDynamicScheduling){
            if(total < work.numOfElements){
                // Something was left out, assign it to first thread
                m_batchSizes.begin()->second += work.numOfElements - total;
            }else if(total > work.numOfElements){
                // More elements were given, remove from the first threads
                for(auto& ct : threads){
                    // If thread i has enough elements, remove all the extra from it
                    if(m_batchSizes[ct.getId()] > total - work.numOfElements){
                        m_batchSizes[ct.getId()] -= total - work.numOfElements;
                        total = work.numOfElements;
                        break;
                    }

                    // else assign 0 elements to it and move to the next thread
                    total -= m_batchSizes[ct.getId()];
                    m_batchSizes[ct.getId()] = 0;
                }
            }
        }

		#ifdef DBG_RATIO
            for(const auto& pair : m_batchSizes){
                log("Thread %d -> Assigning batch size = %lu elements", pair.first, pair.second);
            }
		#endif

        m_globalFirst      = work.startPoint;
        m_globalLast       = work.startPoint + work.numOfElements - 1;
        m_globalBatchStart = work.startPoint;

		#ifdef DBG_DATA
            log("Got job with globalFirst = %lu, globalLast = %lu", m_globalFirst, m_globalLast);
		#endif

		#ifdef DBG_TIME
			sw.stop();
			time_assign = sw.getMsec();
			sw.start();
		#endif

		// Reset the global listIndex counter
        m_listIndex = 0;

		// Start all the worker threads
        for(auto& cti : threads){
            cti.dispatch(workDispatcher, resultPtrs[cti.getId()], &m_listIndex);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_start = sw.getMsec();
			sw.start();
		#endif

		#ifdef DBG_MPI_STEPS
            log("Waiting for results from %lu threads...", threads.size());
		#endif
		// Wait for all worker threads to finish their work
        for(auto& ct : threads){
            ct.wait();
		}

		#ifdef DBG_TIME
			sw.stop();
			time_wait = sw.getMsec();
		#endif

		// Sanity check
        size_t totalCalculated = 0;
        for(const auto& ct : threads)
            totalCalculated += ct.getLastCalculatedElements();

		if(totalCalculated != work.numOfElements){
            log("Shit happened: Total calculated elements are %lu but %lu were assigned to this slave", totalCalculated, work.numOfElements);

            for(const auto& ct : threads){
                log("Thread %d calculated %lu elements", ct.getId(), ct.getLastCalculatedElements());
            }

			exit(-123);
		}

        #ifdef DBG_RESULTS
            if(m_config.resultSaveType == SAVE_TYPE_LIST){
                log("Found %u results", m_listIndex / m_config.model.D);
            }
        #endif
        #ifdef DBG_RESULTS_RAW
            if(m_config.resultSaveType == SAVE_TYPE_ALL){
                std::string str;
                str += "[ ";
                for(unsigned long i=0; i<work.numOfElements; i++){
                    char tmp[64];
                    sprintf(tmp, "%f ", m_results[i]);
                    str += tmp;
				}
                str += "]";
                log("%s", str.c_str());
            }else{
                std::string str;
                for(int i=0; i<m_listIndex; i+=m_config.model.D){
                    str += "[ ";
                    for(unsigned int j=0; j<m_config.model.D; j++){
                        char tmp[64];
                        sprintf(tmp, "%f ", ((DATA_TYPE *) m_results.data())[i + j]);
                        str += tmp;
                    }
                    str += " ]";
                }
                log("Results (m_listIndex = %d): %s", m_listIndex, str.c_str());
			}
		#endif

        #ifdef DBG_TIME
            sw.start();
        #endif
        if(m_config.threadBalancing){

			bool failed = false;
			float totalScore = 0;
			float newRatio;
            std::map<ComputeThreadID, float> scores;

            for(const auto& cti : threads){
                float runTime = cti.getLastRunTime();
                if(runTime < 0){
                    log("Error: Skipping ratio correction due to negative time");
					failed = true;
					break;
				}

                if(runTime < m_config.minMsForRatioAdjustment){
					#ifdef DBG_RATIO
                        log("Skipping ratio correction due to fast execution");
					#endif

					failed = true;
					break;
				}

                if(m_config.benchmark){
                    log("Thread %d time: %f ms", cti.getId(), runTime);
				}

                scores[cti.getId()] = cti.getLastCalculatedElements() / runTime;
                totalScore += scores[cti.getId()];
			}

			if(!failed){
				// Adjust the ratio for each thread
				numOfRatioAdjustments++;
                for(auto& ct : threads){
                    newRatio = scores[ct.getId()] / totalScore;
                    ratioSums[ct.getId()] += newRatio;

                    if(m_config.threadBalancingAverage){
                        ratios[ct.getId()] = ratioSums[ct.getId()] / numOfRatioAdjustments;
					}else{
                        ratios[ct.getId()] = newRatio;
					}

					#ifdef DBG_RATIO
                        log("Adjusting thread %d ratio to %f (elements = %lu, time = %f ms, score = %f)",
                                ct.getId(), ratios[ct.getId()], ct.getLastCalculatedElements(),
                                ct.getLastRunTime(), ct.getLastCalculatedElements()/ct.getLastRunTime());
					#endif
				}
            }
		}
        #ifdef DBG_TIME
            sw.stop();
            time_scores = sw.getMsec();
            sw.start();
        #endif

		// Send all results to master
		#ifdef DBG_MPI_STEPS
            log("Sending data to master...");
		#endif

		#ifdef DBG_TIME
			sw.start();
		#endif

        if(m_config.resultSaveType == SAVE_TYPE_ALL){
            DesFramework::sendResults(m_results.data(), work.numOfElements);
        }else{
            DesFramework::sendListResults((DATA_TYPE*) m_results.data(), m_listIndex / m_config.model.D, m_config.model.D);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_results = sw.getMsec();

            log("Benchmark:");
            log("Time for receiving data: %f ms", time_data);
            log("Time for assign: %f ms", time_assign);
            log("Time for start: %f ms", time_start);
            log("Time for wait: %f ms", time_wait);
            log("Time for scores: %f ms", time_scores);
            log("Time for results: %f ms", time_results);
		#endif
    }
}

unsigned long CoordinatorThread::calculateMaxBatchSize(const DesConfig& config)
{
    unsigned long maxBatchSize;
    if(config.output.overrideMemoryRestrictions){
        maxBatchSize = config.batchSize;
    }else{
        if(config.processingType == PROCESSING_TYPE_CPU)
            maxBatchSize = getMaxCPUBytes();
        else if(config.processingType == PROCESSING_TYPE_GPU)
            maxBatchSize = getMaxGPUBytes();
        else
            maxBatchSize = std::min(getMaxCPUBytes(), getMaxGPUBytes());

        // maxBatchSize contains the value in bytes, so divide it according to resultSaveType to convert it to actual batch size
        if(config.resultSaveType == SAVE_TYPE_ALL)
            maxBatchSize /= sizeof(RESULT_TYPE);
        else
            maxBatchSize /= config.model.D * sizeof(DATA_TYPE);

        // Limit the batch size by the user-given value
        maxBatchSize = std::min((unsigned long)config.batchSize, (unsigned long)maxBatchSize);

        // If we are saving a list, the max number of elements we might want to send is maxBatchSize * D, so limit the batch size
        // so that the max number of elements is INT_MAX
        if(config.resultSaveType == SAVE_TYPE_LIST && (unsigned long) (maxBatchSize*config.model.D) > (unsigned long) INT_MAX){
            maxBatchSize = (INT_MAX - config.model.D) / config.model.D;
        }
        // If we are saving all of the results, the max number of elements is maxBatchSize itself, so limit it to INT_MAX
        else if(config.resultSaveType == SAVE_TYPE_ALL && (unsigned long) maxBatchSize > (unsigned long) INT_MAX){
            maxBatchSize = INT_MAX;
        }
    }

    return maxBatchSize;
}

unsigned long CoordinatorThread::calculateMaxCpuBatchSize(const DesConfig& config)
{
    unsigned long maxCpuElements = getMaxCPUBytes();
    if(config.resultSaveType == SAVE_TYPE_ALL) {
        maxCpuElements /= sizeof(RESULT_TYPE);
    } else {
        maxCpuElements /= config.model.D * sizeof(DATA_TYPE);
    }

    return maxCpuElements;
}

}
