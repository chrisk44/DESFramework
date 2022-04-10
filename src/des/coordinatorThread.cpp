#include "desf.h"

#include "utilities.h"

void DesFramework::coordinatorThread(std::vector<ComputeThread>& computeThreads, ThreadCommonData& tcd){
    int* globalListIndexPtr = &tcd.listIndex;

	AssignedWork work;
	unsigned long maxBatchSize;
	unsigned long allocatedElements = 0;
    RESULT_TYPE* localResults = nullptr;
	int numOfRatioAdjustments = 0;
	unsigned long maxCpuElements = getMaxCPUBytes();
    if(m_config.resultSaveType == SAVE_TYPE_ALL)
		maxCpuElements /= sizeof(RESULT_TYPE);
	else
        maxCpuElements /= m_config.model.D * sizeof(DATA_TYPE);

    std::map<ComputeThreadID, float> ratios;
    std::map<ComputeThreadID, float> ratioSums;
    for(const auto& ct : computeThreads){
        ratios[ct.getId()] = 1.f / (float) computeThreads.size();
        ratioSums[ct.getId()] = 0.f;
    }

	#ifdef DBG_TIME
		Stopwatch sw;
		float time_data, time_assign, time_start, time_wait, time_scores, time_results;
	#endif

    if(m_config.output.overrideMemoryRestrictions){
        maxBatchSize = m_config.batchSize;
	}else{
        if(m_config.processingType == PROCESSING_TYPE_CPU)
			maxBatchSize = getMaxCPUBytes();
        else if(m_config.processingType == PROCESSING_TYPE_GPU)
			maxBatchSize = getMaxGPUBytes();
		else
            maxBatchSize = std::min(getMaxCPUBytes(), getMaxGPUBytes());

		// maxBatchSize contains the value in bytes, so divide it according to resultSaveType to convert it to actual batch size
        if(m_config.resultSaveType == SAVE_TYPE_ALL)
            maxBatchSize /= sizeof(RESULT_TYPE);
        else
            maxBatchSize /= m_config.model.D * sizeof(DATA_TYPE);

		// Limit the batch size by the user-given value
        maxBatchSize = std::min((unsigned long)m_config.batchSize, (unsigned long)maxBatchSize);

		// If we are saving a list, the max number of elements we might want to send is maxBatchSize * D, so limit the batch size
		// so that the max number of elements is INT_MAX
        if(m_config.resultSaveType == SAVE_TYPE_LIST && (unsigned long) (maxBatchSize*m_config.model.D) > (unsigned long) INT_MAX){
            maxBatchSize = (INT_MAX - m_config.model.D) / m_config.model.D;
		}
		// If we are saving all of the results, the max number of elements is maxBatchSize itself, so limit it to INT_MAX
        else if(m_config.resultSaveType == SAVE_TYPE_ALL && (unsigned long) maxBatchSize > (unsigned long) INT_MAX){
			maxBatchSize = INT_MAX;
		}
	}

	#ifdef DBG_START_STOP
        printf("[%d] Coordinator: Max batch size: %lu\n", m_rank, maxBatchSize);
	#endif

	while(true){
		// Send READY signal to master
		#ifdef DBG_MPI_STEPS
            printf("[%d] Coordinator: Sending READY...\n", m_rank);
		#endif
		#ifdef DBG_TIME
			sw.start();
		#endif

        // Get a batch of data from master
        sendReadyRequest(maxBatchSize);
        work = receiveWorkFromMaster();

		#ifdef DBG_DATA
            printf("[%d] Coordinator: Received %lu elements starting from %lu\n", m_rank, work.numOfElements, work.startPoint);
		#endif
		#ifdef DBG_MPI_STEPS
            printf("[%d] Coordinator: Received %lu elements\n", m_rank, work.numOfElements);
		#endif

		// If no elements, break
		if(work.numOfElements == 0)
			break;

		// Make sure we have enough allocated memory for localResults
		if(work.numOfElements > allocatedElements && allocatedElements < maxCpuElements){
			#ifdef DBG_MEMORY
                printf("[%d] Coordinator: Allocating more memory for localResults: %lu (%p) -> ", m_rank, allocatedElements, localResults);
			#endif

            allocatedElements = std::min(work.numOfElements, maxCpuElements);
			localResults = (RESULT_TYPE*) realloc(localResults, allocatedElements * sizeof(RESULT_TYPE));
			if(localResults == nullptr){
				fatal("Can't allocate memory for localResults");
			}

            tcd.results = localResults;

			#ifdef DBG_MEMORY
                printf("%lu (%p)\n", allocatedElements, localResults);
			#endif
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
        std::map<ComputeThreadID, size_t> batchSizes;
        unsigned long total = 0;
        for(auto& ct : computeThreads){
            if(m_config.slaveDynamicScheduling)
                batchSizes[ct.getId()] = std::max((unsigned long) 1, (unsigned long) (ratios[ct.getId()] * std::min(
                    m_config.slaveBatchSize, work.numOfElements)));
            else{
                batchSizes[ct.getId()] = std::max((unsigned long) 1, (unsigned long) (ratios[ct.getId()] * work.numOfElements));
                total += batchSizes[ct.getId()];
            }
        }

        /*
         * If we are NOT using dynamic scheduling, make sure that exactly
         * work.numOfElements elements have been assigned
         */
        if(!m_config.slaveDynamicScheduling){
            if(total < work.numOfElements){
                // Something was left out, assign it to first thread
                batchSizes.begin()->second += work.numOfElements - total;
            }else if(total > work.numOfElements){
                // More elements were given, remove from the first threads
                for(auto& ct : computeThreads){
                    // If thread i has enough elements, remove all the extra from it
                    if(batchSizes[ct.getId()] > total - work.numOfElements){
                        batchSizes[ct.getId()] -= total - work.numOfElements;
                        total = work.numOfElements;
                        break;
                    }

                    // else assign 0 elements to it and move to the next thread
                    total -= batchSizes[ct.getId()];
                    batchSizes[ct.getId()] = 0;
                }
            }
        }

		#ifdef DBG_RATIO
            for(const auto& pair : batchSizes){
                printf("[%d] Coordinator: Thread %d -> Assigning batch size = %lu elements\n", m_rank, pair.first, pair.second);
            }
		#endif

        tcd.globalFirst         = work.startPoint;
        tcd.globalLast			= work.startPoint + work.numOfElements - 1;
        tcd.globalBatchStart	= work.startPoint;

		#ifdef DBG_DATA
			printf("[%d] Coordinator: Got job with globalFirst = %lu, globalLast = %lu\n",
                            m_rank, tcd.globalFirst, tcd.globalLast);
		#endif

		#ifdef DBG_TIME
			sw.stop();
			time_assign = sw.getMsec();
			sw.start();
		#endif

		// Reset the global listIndex counter
		*globalListIndexPtr = 0;

		// Start all the worker threads
        for(auto& cti : computeThreads){
            cti.dispatch(batchSizes[cti.getId()]);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_start = sw.getMsec();
			sw.start();
		#endif

		#ifdef DBG_MPI_STEPS
            printf("[%d] Coordinator: Waiting for results from %lu threads...\n", m_rank, computeThreads.size());
		#endif
		// Wait for all worker threads to finish their work
        for(auto& ct : computeThreads){
            ct.wait();
		}

		#ifdef DBG_TIME
			sw.stop();
			time_wait = sw.getMsec();
		#endif

		// Sanity check
        size_t totalCalculated = 0;
        for(const auto& ct : computeThreads)
            totalCalculated += ct.getLastCalculatedElements();

		if(totalCalculated != work.numOfElements){
			printf("[%d] Coordinator: Shit happened. Total calculated elements are %lu but %lu were assigned to this slave\n",
                                m_rank, totalCalculated, work.numOfElements);

            for(const auto& ct : computeThreads){
                printf("[%d] Coordinator: Thread %d calculated %lu elements\n", m_rank, ct.getId(), ct.getLastCalculatedElements());
            }

			exit(-123);
		}

        #ifdef DBG_RESULTS
            if(m_config.resultSaveType == SAVE_TYPE_LIST){
                printf("[%d] Coordinator: Found %u results\n", m_rank, *globalListIndexPtr / m_config.model.D);
            }
        #endif
        #ifdef DBG_RESULTS_RAW
            if(m_config.resultSaveType == SAVE_TYPE_ALL){
                printf("[%d] Coordinator: Results: [", m_rank);
                for(unsigned long i=0; i<work.numOfElements; i++){
					printf("%f ", localResults[i]);
				}
				printf("]\n");
            }else{
                printf("[%d] Coordinator: Results (*globalListIndexPtr = %d):", m_rank, *globalListIndexPtr);
                for(int i=0; i<*globalListIndexPtr; i+=m_config.model.D){
                    printf("[ ");
                    for(unsigned int j=0; j<m_config.model.D; j++){
                        printf("%f ", ((DATA_TYPE *)localResults)[i + j]);
                    }
                    printf("]");
                }
                printf("\n");
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

            for(const auto& cti : computeThreads){
                float runTime = cti.getLastRunTime();
                if(runTime < 0){
                    printf("[%d] Coordinator: Error: Skipping ratio correction due to negative time\n", m_rank);
					failed = true;
					break;
				}

                if(runTime < m_config.minMsForRatioAdjustment){
					#ifdef DBG_RATIO
                        printf("[%d] Coordinator: Skipping ratio correction due to fast execution\n", m_rank);
					#endif

					failed = true;
					break;
				}

                if(m_config.benchmark){
                    printf("[%d] Coordinator: Thread %d time: %f ms\n", m_rank, cti.getId(), runTime);
				}

                scores[cti.getId()] = cti.getLastCalculatedElements() / runTime;
                totalScore += scores[cti.getId()];
			}

			if(!failed){
				// Adjust the ratio for each thread
				numOfRatioAdjustments++;
                for(auto& ct : computeThreads){
                    newRatio = scores[ct.getId()] / totalScore;
                    ratioSums[ct.getId()] += newRatio;

                    if(m_config.threadBalancingAverage){
                        ratios[ct.getId()] = ratioSums[ct.getId()] / numOfRatioAdjustments;
					}else{
                        ratios[ct.getId()] = newRatio;
					}

					#ifdef DBG_RATIO
                        printf("[%d] Coordinator: Adjusting thread %d ratio to %f (elements = %lu, time = %f ms, score = %f)\n",
                                m_rank, ct.getId(), ratios[ct.getId()], ct.getLastCalculatedElements(),
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
            printf("[%d] Coordinator: Sending data to master...\n", m_rank);
		#endif

		#ifdef DBG_TIME
			sw.start();
		#endif

        if(m_config.resultSaveType == SAVE_TYPE_ALL){
            sendResults(localResults, work.numOfElements);
		}else{
            sendListResults((DATA_TYPE*) localResults, (*globalListIndexPtr) / m_config.model.D);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_results = sw.getMsec();

            printf("[%d] Coordinator: Benchmark:\n", m_rank);
			printf("Time for receiving data: %f ms\n", time_data);
			printf("Time for assign: %f ms\n", time_assign);
			printf("Time for start: %f ms\n", time_start);
			printf("Time for wait: %f ms\n", time_wait);
			printf("Time for scores: %f ms\n", time_scores);
			printf("Time for results: %f ms\n", time_results);
		#endif
    }

	// Signal worker threads to finish
    tcd.globalFirst = 1;
    tcd.globalLast = 0;
//    for(auto& ct : computeThreads){
////        cti.postStartSemaphore();
//        ct.stop();
//	}

	free(localResults);
}

