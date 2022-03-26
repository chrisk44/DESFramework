#include "framework.h"

void ParallelFramework::coordinatorThread(std::vector<ComputeThreadInfo>& computeThreadInfo, ThreadCommonData& tcd){
    int* globalListIndexPtr = &tcd.listIndex;

	AssignedWork work;
	unsigned long maxBatchSize;
	unsigned long allocatedElements = 0;
    RESULT_TYPE* localResults = nullptr;
	int numOfRatioAdjustments = 0;
	unsigned long maxCpuElements = getMaxCPUBytes();
    if(parameters.resultSaveType == SAVE_TYPE_ALL)
		maxCpuElements /= sizeof(RESULT_TYPE);
	else
        maxCpuElements /= parameters.D * sizeof(DATA_TYPE);

	#ifdef DBG_TIME
		Stopwatch sw;
		float time_data, time_assign, time_start, time_wait, time_scores, time_results;
	#endif

    if(parameters.overrideMemoryRestrictions){
        maxBatchSize = parameters.batchSize;
	}else{
        if(parameters.processingType == PROCESSING_TYPE_CPU)
			maxBatchSize = getMaxCPUBytes();
        else if(parameters.processingType == PROCESSING_TYPE_GPU)
			maxBatchSize = getMaxGPUBytes();
		else
            maxBatchSize = std::min(getMaxCPUBytes(), getMaxGPUBytes());

		// maxBatchSize contains the value in bytes, so divide it according to resultSaveType to convert it to actual batch size
        if(parameters.resultSaveType == SAVE_TYPE_ALL)
            maxBatchSize /= sizeof(RESULT_TYPE);
        else
            maxBatchSize /= parameters.D * sizeof(DATA_TYPE);

		// Limit the batch size by the user-given value
        maxBatchSize = std::min((unsigned long)parameters.batchSize, (unsigned long)maxBatchSize);

		// If we are saving a list, the max number of elements we might want to send is maxBatchSize * D, so limit the batch size
		// so that the max number of elements is INT_MAX
        if(parameters.resultSaveType == SAVE_TYPE_LIST && (unsigned long) (maxBatchSize*parameters.D) > (unsigned long) INT_MAX){
            maxBatchSize = (INT_MAX - parameters.D) / parameters.D;
		}
		// If we are saving all of the results, the max number of elements is maxBatchSize itself, so limit it to INT_MAX
        else if(parameters.resultSaveType == SAVE_TYPE_ALL && (unsigned long) maxBatchSize > (unsigned long) INT_MAX){
			maxBatchSize = INT_MAX;
		}
	}

	#ifdef DBG_START_STOP
		printf("[%d] Coordinator: Max batch size: %lu\n", rank, maxBatchSize);
	#endif

	while(true){
		// Send READY signal to master
		#ifdef DBG_MPI_STEPS
			printf("[%d] Coordinator: Sending READY...\n", rank);
		#endif
		#ifdef DBG_TIME
			sw.start();
		#endif

        // Get a batch of data from master
        sendReadyRequest(maxBatchSize);
        work = receiveWorkFromMaster();

		#ifdef DBG_DATA
			printf("[%d] Coordinator: Received %lu elements starting from %lu\n", rank, work.numOfElements, work.startPoint);
		#endif
		#ifdef DBG_MPI_STEPS
			printf("[%d] Coordinator: Received %lu elements\n", rank, work.numOfElements);
		#endif

		// If no elements, break
		if(work.numOfElements == 0)
			break;

		// Make sure we have enough allocated memory for localResults
		if(work.numOfElements > allocatedElements && allocatedElements < maxCpuElements){
			#ifdef DBG_MEMORY
				printf("[%d] Coordinator: Allocating more memory for localResults: %lu (0x%x) -> ", rank, allocatedElements, localResults);
			#endif

            allocatedElements = std::min(work.numOfElements, maxCpuElements);
			localResults = (RESULT_TYPE*) realloc(localResults, allocatedElements * sizeof(RESULT_TYPE));
			if(localResults == nullptr){
				fatal("Can't allocate memory for localResults");
			}

            tcd.results = localResults;

			#ifdef DBG_MEMORY
				printf("%lu (0x%x)\n", allocatedElements, localResults);
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
		unsigned long total = 0;
        for(auto& cti : computeThreadInfo){
            if(parameters.slaveDynamicScheduling)
                cti.batchSize = std::max((unsigned long) 1, (unsigned long) (cti.ratio * std::min(
                    parameters.slaveBatchSize, work.numOfElements)));
			else{
                cti.batchSize = std::max((unsigned long) 1, (unsigned long) (cti.ratio * work.numOfElements));
                total += cti.batchSize;
			}
		}

		/*
		 * If we are NOT using dynamic scheduling, make sure that exactly
		 * work.numOfElements elements have been assigned
		 */
        if(!parameters.slaveDynamicScheduling){
			if(total < work.numOfElements){
				// Something was left out, assign it to first thread
                computeThreadInfo.begin()->batchSize += work.numOfElements - total;
			}else if(total > work.numOfElements){
				// More elements were given, remove from the first threads
                for(auto& cti : computeThreadInfo){
					// If thread i has enough elements, remove all the extra from it
                    if(cti.batchSize > total - work.numOfElements){
                        cti.batchSize -= total - work.numOfElements;
						total = work.numOfElements;
						break;
					}

					// else assign 0 elements to it and move to the next thread
                    total -= cti.batchSize;
                    cti.batchSize = 0;
				}
			}
		}

		#ifdef DBG_RATIO
            for(auto& pair : computeThreadInfo){
				printf("[%d] Coordinator: Thread %d -> Assigning batch size = %lu elements\n",
                        rank, pair.first, pair.second.batchSize);
            }
		#endif

        tcd.globalFirst         = work.startPoint;
        tcd.globalLast			= work.startPoint + work.numOfElements - 1;
        tcd.globalBatchStart	= work.startPoint;

		#ifdef DBG_DATA
			printf("[%d] Coordinator: Got job with globalFirst = %lu, globalLast = %lu\n",
                            rank, tcd.globalFirst, tcd.globalLast);
		#endif

		#ifdef DBG_TIME
			sw.stop();
			time_assign = sw.getMsec();
			sw.start();
		#endif

		// Reset the global listIndex counter
		*globalListIndexPtr = 0;

		// Start all the worker threads
        for(auto& cti : computeThreadInfo){
            cti.elementsCalculated = 0;
            cti.postStartSemaphore();
		}

		#ifdef DBG_TIME
			sw.stop();
			time_start = sw.getMsec();
			sw.start();
		#endif

		#ifdef DBG_MPI_STEPS
            printf("[%d] Coordinator: Waiting for results from %d threads...\n", rank, numOfThreads);
		#endif
		// Wait for all worker threads to finish their work
        for(size_t i=0; i<computeThreadInfo.size(); i++){
            tcd.waitResultsSemaphore();
		}

		#ifdef DBG_TIME
			sw.stop();
			time_wait = sw.getMsec();
		#endif

		// Sanity check
		unsigned long totalCalculated = 0;
        for(const auto& cti : computeThreadInfo)
            totalCalculated += cti.elementsCalculated;

		if(totalCalculated != work.numOfElements){
			printf("[%d] Coordinator: Shit happened. Total calculated elements are %lu but %lu were assigned to this slave\n",
								rank, totalCalculated, work.numOfElements);

			exit(-123);
		}

		#ifdef DBG_RESULTS
            if(parameters.resultSaveType == SAVE_TYPE_ALL){
				printf("[%d] Coordinator: Results: [", rank);
                for(unsigned long i=0; i<work.numOfElements; i++){
					printf("%f ", localResults[i]);
				}
				printf("]\n");
			}else{
				printf("[%d] Coordinator: Results (*globalListIndexPtr = %d):", rank, *globalListIndexPtr);
                for(int i=0; i<*globalListIndexPtr; i+=parameters.D){
					printf("[ ");
                    for(unsigned int j=0; j<parameters.D; j++){
						printf("%f ", ((DATA_TYPE *)localResults)[i + j]);
					}
					printf("]");
				}
				printf("\n");
			}
		#endif

        if(parameters.threadBalancing){
			#ifdef DBG_TIME
				sw.start();
			#endif

			bool failed = false;
			float totalScore = 0;
			float newRatio;
            std::map<int, float> scores;

            for(const auto& cti : computeThreadInfo){
                if(cti.stopwatch.getMsec() < 0){
					printf("[%d] Coordinator: Error: Skipping ratio correction due to negative time\n", rank);
					failed = true;
					break;
				}

                if(cti.stopwatch.getMsec() < parameters.minMsForRatioAdjustment){
					#ifdef DBG_RATIO
						printf("[%d] Coordinator: Skipping ratio correction due to fast execution\n", rank);
					#endif

					failed = true;
					break;
				}

                if(parameters.benchmark){
                    printf("[%d] Coordinator: Thread %d time: %f ms\n", rank, cti.id, cti.stopwatch.getMsec());
				}

                scores[cti.id] = cti.elementsCalculated / cti.stopwatch.getMsec();
                totalScore += scores[cti.id];
			}

			if(!failed){
				// Adjust the ratio for each thread
				numOfRatioAdjustments++;
                for(auto& cti : computeThreadInfo){
                    newRatio = scores[cti.id] / totalScore;
                    cti.totalRatio += newRatio;

                    if(parameters.threadBalancingAverage){
                        cti.ratio = cti.totalRatio / numOfRatioAdjustments;
					}else{
                        cti.ratio = newRatio;
					}

					#ifdef DBG_RATIO
                        printf("[%d] Coordinator: Adjusting thread %d ratio to %f (elements = %lu, time = %f ms, score = %f)\n",
                                rank, cti.id, cti.ratio, cti.elementsCalculated,
                                cti.stopwatch.getMsec(), cti.elementsCalculated/cti.stopwatch.getMsec());
					#endif
				}
			}

			#ifdef DBG_TIME
				sw.stop();
				time_scores = sw.getMsec();
				sw.start();
			#endif
		}

		// Send all results to master
		#ifdef DBG_MPI_STEPS
			printf("[%d] Coordinator: Sending data to master...\n", rank);
		#endif

		#ifdef DBG_TIME
			sw.start();
		#endif

        if(parameters.resultSaveType == SAVE_TYPE_ALL){
            sendResults(localResults, work.numOfElements);
		}else{
            sendListResults((DATA_TYPE*) localResults, (*globalListIndexPtr) / parameters.D);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_results = sw.getMsec();

			printf("[%d] Coordinator: Benchmark:\n", rank);
			printf("Time for receiving data: %f ms\n", time_data);
			printf("Time for assign: %f ms\n", time_assign);
			printf("Time for start: %f ms\n", time_start);
			printf("Time for wait: %f ms\n", time_wait);
			printf("Time for scores: %f ms\n", time_scores);
			printf("Time for results: %f ms\n", time_results);
		#endif
	}

    // Notify about exiting
    sendExitSignal();

	// Signal worker threads to finish
    tcd.globalFirst = 1;
    tcd.globalLast = 0;
    for(auto& cti : computeThreadInfo){
        cti.postStartSemaphore();
	}

	free(localResults);
}

